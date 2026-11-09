"""Машина состояний стола, профиль `restaurant` (SPEC §5 + §9a).

FREE -> SEATED_WAITING -> SERVED -> NEEDS_CLEANING -> FREE

Чистая логика: на вход ZoneObservation, на выход Event. Никакого видео, id треков используются
только для эпизодов визитов; занятость — по скользящему окну присутствия.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from qonaq.config import AlertThresholds, RestaurantThresholds
from qonaq.logic.base import Event, PresenceFilter, ZoneObservation


class TableState(StrEnum):
    FREE = "FREE"
    SEATED_WAITING = "SEATED_WAITING"
    SERVED = "SERVED"
    NEEDS_CLEANING = "NEEDS_CLEANING"


OCCUPIED = (TableState.SEATED_WAITING, TableState.SERVED)


@dataclass
class _Episode:
    """Эпизод «у стола кто-то не из гостей» — кандидат в визит персонала."""

    start: float
    last_seen: float
    first_seen: dict[int, float] = field(default_factory=dict)  # id -> когда появился
    has_staff: bool = False
    max_single: float = 0.0  # самое долгое пребывание одного id
    confirmed: bool = False  # визит уже засчитан (для подтверждённого персонала — сразу)
    emitted: bool = False  # по визиту уже выпущено событие (served / staff_visit / cleaned)

    @property
    def duration(self) -> float:
        return self.last_seen - self.start


@dataclass
class _Session:
    sid: int
    seated_ts: float
    preexisting: bool = False
    dirty_seat: bool = False
    served_ts: float | None = None
    last_visit_ts: float | None = None
    visits: int = 0
    max_guests: int = 0
    low_confidence: bool = False
    uses_any: bool = False
    alerts: set[str] = field(default_factory=set)


class _PosLike(Protocol):  # минимальный интерфейс POS, нужен только для алерта «занят без чека»
    def has_open_check(self, zone_id: str, ts: float) -> bool: ...


class TableMachine:
    def __init__(
        self,
        zone_id: str,
        th: RestaurantThresholds | None = None,
        alerts: AlertThresholds | None = None,
        pos: _PosLike | None = None,
    ):
        self.zone_id = zone_id
        self.th = th or RestaurantThresholds()
        self.al = alerts or AlertThresholds()
        self.pos = pos
        self.state = TableState.FREE
        self.state_since = 0.0
        w, on, off = self.th.dropout_tolerance_s, self.th.presence_on_ratio, self.th.presence_off_ratio
        self._any = PresenceFilter(w, on, off)  # гости + неизвестные (не персонал)
        self._guest = PresenceFilter(w, on, off)  # только подтверждённые гости
        self._first_ts: float | None = None
        self._ts = 0.0
        self._session: _Session | None = None
        self._sid = 0
        self._episode: _Episode | None = None
        self._absent_visits: list[_Episode] = []  # визиты, пока гостей нет на месте (уборка или визит?)
        self._left_ts: float | None = None
        self._dirty_alerted = False
        self._clean_since: float | None = None
        self._n_guest = 0
        self._n_unknown = 0
        self._n_staff = 0

    # ------------------------------------------------------------------ helpers
    def _ev(self, ts: float, type_: str, **payload: Any) -> Event:
        payload.setdefault("detected_ts", round(self._ts, 3))
        if self._sid:
            payload.setdefault("sid", self._sid)
        return Event(ts=ts, zone_id=self.zone_id, type=type_, payload=payload)

    def _set_state(self, new: TableState, ts: float, out: list[Event]) -> None:
        if new == self.state:
            return
        out.append(self._ev(ts, "state_change", **{"from": self.state.value, "to": new.value}))
        self.state = new
        self.state_since = ts

    def _occupancy(self) -> PresenceFilter:
        s = self._session
        return self._any if (s is not None and s.uses_any) else self._guest

    # ------------------------------------------------------------------ main
    def update(self, obs: ZoneObservation) -> list[Event]:
        ts = obs.ts
        self._ts = ts
        if self._first_ts is None:
            self._first_ts = ts
            self.state_since = ts
        out: list[Event] = []
        guests, staff = set(obs.guest_tracks), set(obs.staff_tracks)
        unknown = set(obs.unknown_tracks) - guests - staff
        self._n_guest, self._n_unknown, self._n_staff = len(guests), len(unknown), len(staff)
        self._any.update(ts, bool(guests or unknown))
        self._guest.update(ts, bool(guests))
        if self._session is not None:
            self._session.max_guests = max(self._session.max_guests, len(guests))

        self._update_episode(ts, guests, staff, unknown, out)

        if self.state == TableState.FREE:
            self._try_seat(ts, out, dirty=False)
        elif self.state == TableState.SEATED_WAITING:
            self._step_waiting(ts, out)
        elif self.state == TableState.SERVED:
            self._step_served(ts, obs, out)
        elif self.state == TableState.NEEDS_CLEANING:
            self._step_dirty(ts, obs, out)
        self._check_pos(ts, out)
        return out

    # ------------------------------------------------------------------ visits
    def _update_episode(self, ts: float, guests: set[int], staff: set[int], unknown: set[int],
                        out: list[Event]) -> None:
        if self.state == TableState.FREE:
            # У свободного стола неизвестные — кандидаты в гости, персонал не интересен.
            self._episode = None
            return
        ep = self._episode
        if ep is not None:
            # Неизвестный, которого роли признали гостем, — не визит персонала.
            for tid in list(ep.first_seen):
                if tid in guests:
                    del ep.first_seen[tid]
            if not ep.first_seen and not ep.confirmed:
                ep = self._episode = None
        visitors = staff | unknown
        if visitors:
            if ep is None:
                ep = self._episode = _Episode(start=ts, last_seen=ts)
            for tid in visitors:
                ep.first_seen.setdefault(tid, ts)
                ep.max_single = max(ep.max_single, ts - ep.first_seen[tid])
            ep.last_seen = ts
            ep.has_staff = ep.has_staff or bool(staff)
            # Подтверждённый персонал: визит засчитывается сразу по достижении порога.
            need = self.th.clean_min_s if self.state == TableState.NEEDS_CLEANING else self.th.visit_min_s
            if ep.has_staff and not ep.confirmed and ep.duration >= need:
                ep.confirmed = True
                self._on_visit(ep, ts, out, final=False)
        elif ep is not None and ts - ep.last_seen > self.th.visit_gap_s:
            self._episode = None
            self._finish_episode(ep, ts, out)

    def _finish_episode(self, ep: _Episode, ts: float, out: list[Event]) -> None:
        if ep.confirmed:
            if not ep.emitted and self.state == TableState.SERVED and self._session is not None:
                if self._guest_here():
                    self._emit_staff_visit(ep, out)
                else:
                    self._absent_visits.append(ep)
            return
        th = self.th
        if ep.has_staff:
            return  # персонал, но короче порога — не визит
        dirty = self.state == TableState.NEEDS_CLEANING
        need = th.clean_min_s if dirty else th.visit_min_s
        upper = float("inf") if dirty else th.visit_max_s
        if need <= ep.duration <= upper and ep.max_single >= need / 2:
            ep.confirmed = True
            self._on_visit(ep, ts, out, final=True)

    def _guest_here(self) -> bool:
        return self._occupancy().state

    def _on_visit(self, ep: _Episode, ts: float, out: list[Event], final: bool) -> None:
        conf = "high" if ep.has_staff else "low"
        s = self._session
        if self.state == TableState.SEATED_WAITING and s is not None:
            s.served_ts = ep.start
            s.last_visit_ts = ep.last_seen
            s.visits = 1
            s.low_confidence = s.low_confidence or not ep.has_staff
            ep.emitted = True
            out.append(self._ev(ep.start, "served", response_time_s=round(ep.start - s.seated_ts, 1),
                                confidence=conf))
            self._set_state(TableState.SERVED, ep.start, out)
        elif self.state == TableState.SERVED and s is not None:
            if not final:
                return  # повторный визит фиксируем по его окончании — с полной длительностью
            if not self._guest_here():
                # Гостей сейчас нет: это может быть уборка после ухода. Решим, когда станет ясно.
                self._absent_visits.append(ep)
                return
            self._emit_staff_visit(ep, out)
        elif self.state == TableState.NEEDS_CLEANING:
            ep.emitted = True
            self._cleaned(ep.last_seen if final else ts, out, by="staff_visit", confidence=conf)

    def _emit_staff_visit(self, ep: _Episode, out: list[Event]) -> None:
        s = self._session
        assert s is not None
        ep.emitted = True
        s.visits += 1
        s.last_visit_ts = ep.last_seen
        s.alerts.discard("long_service_without_visit")
        out.append(self._ev(ep.start, "staff_visit", duration_s=round(ep.duration, 1), n=s.visits,
                            confidence="high" if ep.has_staff else "low"))

    # ------------------------------------------------------------------ states
    def _try_seat(self, ts: float, out: list[Event], dirty: bool) -> bool:
        th = self.th
        if not self._any.state:
            return False
        dur = self._any.duration(ts)
        if dur < th.seat_min_s:
            return False
        by_role = self._guest.state or self._n_guest > 0
        if not by_role and dur < th.seat_min_s * th.seat_unknown_factor:
            return False
        seated_ts = self._any.since if self._any.since is not None else ts
        seated_ts = max(seated_ts, self._left_ts if (dirty and self._left_ts is not None) else self.state_since)
        preexisting = (not dirty) and self._first_ts is not None and (
            seated_ts - self._first_ts <= th.preexisting_grace_s
        )
        self._sid += 1
        self._session = _Session(sid=self._sid, seated_ts=seated_ts, preexisting=preexisting,
                                 dirty_seat=dirty, uses_any=not by_role, low_confidence=not by_role,
                                 max_guests=self._n_guest)
        self._episode = None
        self._absent_visits = []
        self._clean_since = None
        if dirty:
            dirty_s = round(seated_ts - self._left_ts, 1) if self._left_ts is not None else None
            out.append(self._ev(seated_ts, "seated_at_dirty_table", dirty_time_s=dirty_s))
        out.append(self._ev(seated_ts, "seated", preexisting=preexisting,
                            guests=max(self._n_guest, 1 if by_role else self._n_unknown),
                            confidence="high" if by_role else "low"))
        if preexisting:
            # Сидели ещё до старта наблюдения: время ожидания неизвестно, считаем стол обслуженным.
            self._session.served_ts = seated_ts
            self._session.last_visit_ts = seated_ts
            self._set_state(TableState.SERVED, seated_ts, out)
        else:
            self._set_state(TableState.SEATED_WAITING, seated_ts, out)
        self._left_ts = None
        self._dirty_alerted = False
        return True

    def _step_waiting(self, ts: float, out: list[Event]) -> None:
        s = self._session
        assert s is not None
        occ = self._occupancy()
        if not occ.state and occ.duration(ts) >= self.th.leave_min_s:
            left_ts = occ.since if occ.since is not None else ts
            out.append(self._ev(left_ts, "walkout_before_service", waited_s=round(left_ts - s.seated_ts, 1)))
            self._set_state(TableState.FREE, left_ts, out)
            self._session = None
            self._episode = None
            return
        waited = ts - s.seated_ts
        if (waited >= self.al.no_waiter_s and "no_waiter" not in s.alerts and occ.state
                and self._episode is None):
            s.alerts.add("no_waiter")
            out.append(self._ev(ts, "alert", kind="no_waiter", waited_s=round(waited, 1)))

    def _step_served(self, ts: float, obs: ZoneObservation, out: list[Event]) -> None:
        s = self._session
        assert s is not None
        occ = self._occupancy()
        if occ.state:
            if self._absent_visits:
                # Гости вернулись (выходили покурить) — визиты в их отсутствие были обычными визитами.
                for ep in self._absent_visits:
                    self._emit_staff_visit(ep, out)
                self._absent_visits = []
            last = s.last_visit_ts if s.last_visit_ts is not None else (s.served_ts or s.seated_ts)
            if (ts - last >= self.al.long_service_without_visit_s
                    and "long_service_without_visit" not in s.alerts):
                s.alerts.add("long_service_without_visit")
                out.append(self._ev(ts, "alert", kind="long_service_without_visit",
                                    since_visit_s=round(ts - last, 1)))
            return
        if occ.duration(ts) < self.th.leave_min_s:
            return
        left_ts = occ.since if occ.since is not None else ts
        out.append(self._ev(left_ts, "left", visit_duration_s=round(left_ts - s.seated_ts, 1),
                            staff_visits=s.visits, preexisting=s.preexisting))
        self._left_ts = left_ts
        self._dirty_alerted = False
        # Уборка могла случиться, пока мы ждали подтверждения ухода (leave_min_s).
        cleaners = [ep for ep in self._absent_visits
                    if ep.last_seen >= left_ts and ep.duration >= self.th.clean_min_s]
        self._absent_visits = []
        if cleaners:
            ep = cleaners[0]
            self._set_state(TableState.NEEDS_CLEANING, left_ts, out)
            self._cleaned(ep.last_seen, out, by="staff_visit",
                          confidence="high" if ep.has_staff else "low")
        elif self.th.use_table_appearance and obs.table_dirty is False:
            self._set_state(TableState.NEEDS_CLEANING, left_ts, out)
            self._cleaned(left_ts, out, by="appearance", confidence="low")
        else:
            self._set_state(TableState.NEEDS_CLEANING, left_ts, out)
            self._session = None

    def _step_dirty(self, ts: float, obs: ZoneObservation, out: list[Event]) -> None:
        if self._try_seat(ts, out, dirty=True):
            return
        if self.th.use_table_appearance and not (self._n_guest or self._n_unknown or self._n_staff):
            if obs.table_dirty is False:
                if self._clean_since is None:
                    self._clean_since = ts
                if ts - self._clean_since >= self.th.appearance_clean_confirm_s:
                    self._cleaned(self._clean_since, out, by="appearance", confidence="low")
                    return
            elif obs.table_dirty is True:
                self._clean_since = None
        if self._left_ts is not None and not self._dirty_alerted:
            dirty = ts - self._left_ts
            if dirty >= self.al.dirty_table_s:
                self._dirty_alerted = True
                out.append(self._ev(ts, "alert", kind="dirty_table", dirty_s=round(dirty, 1)))

    def _cleaned(self, ts: float, out: list[Event], by: str, confidence: str) -> None:
        left = self._left_ts if self._left_ts is not None else self.state_since
        ts = max(ts, left)
        out.append(self._ev(ts, "cleaned", cleanup_time_s=round(ts - left, 1), by=by, confidence=confidence))
        self._set_state(TableState.FREE, ts, out)
        self._session = None
        self._episode = None
        self._left_ts = None
        self._clean_since = None

    def _check_pos(self, ts: float, out: list[Event]) -> None:
        s = self._session
        if self.pos is None or s is None or self.state not in OCCUPIED:
            return
        if "occupied_without_check" in s.alerts or s.preexisting:
            return
        occupied = ts - s.seated_ts
        if occupied >= self.al.occupied_without_check_s and not self.pos.has_open_check(self.zone_id, ts):
            s.alerts.add("occupied_without_check")
            out.append(self._ev(ts, "alert", kind="occupied_without_check", occupied_s=round(occupied, 1)))

    # ------------------------------------------------------------------ view
    def snapshot(self, ts: float | None = None) -> dict[str, Any]:
        ts = self._ts if ts is None else ts
        s = self._session
        snap: dict[str, Any] = {
            "zone_id": self.zone_id,
            "type": "table",
            "state": self.state.value,
            "since": round(self.state_since, 2),
            "state_s": round(max(0.0, ts - self.state_since), 1),
            "guests": self._n_guest if self.state in OCCUPIED else 0,
            "people": self._n_guest + self._n_unknown + self._n_staff,
            "staff_now": self._episode is not None and self._episode.has_staff,
            "visitor_now": self._episode is not None,
            "alert": None,
        }
        if s is not None and self.state in OCCUPIED:
            snap["guests"] = max(self._n_guest, 1) if self._occupancy().state else self._n_guest
            snap["session_s"] = round(ts - s.seated_ts, 1)
            snap["visits"] = s.visits
            snap["preexisting"] = s.preexisting
            if self.state == TableState.SEATED_WAITING:
                snap["wait_s"] = round(ts - s.seated_ts, 1)
                if snap["wait_s"] >= self.al.no_waiter_s:
                    snap["alert"] = "no_waiter"
            elif "long_service_without_visit" in s.alerts:
                snap["alert"] = "long_service_without_visit"
            if "occupied_without_check" in s.alerts:
                snap["alert"] = "occupied_without_check"
        if self.state == TableState.NEEDS_CLEANING and self._left_ts is not None:
            snap["dirty_s"] = round(ts - self._left_ts, 1)
            if snap["dirty_s"] >= self.al.dirty_table_s:
                snap["alert"] = "dirty_table"
        return snap
