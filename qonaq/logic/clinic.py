"""Профиль `clinic` (SPEC §6): зона ожидания + стойка регистрации.

* waiting_area — кто и сколько ждёт (с момента входа трека в зону до перехода к стойке или ухода);
* service_point — очередь у стойки, время обслуживания, алерт «очередь есть, сотрудника нет».

Чистая логика от ZoneObservation. id треков здесь важнее, чем в ресторане (ждём конкретного человека),
поэтому пайплайн подаёт «стабильные» id (с наследованием при смене id трекером), а уход подтверждается
только после exit_min_s отсутствия.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qonaq.config import AlertThresholds, ClinicThresholds, SiteConfig
from qonaq.logic.base import Event, PresenceFilter, ZoneObservation
from qonaq.logic.engine import CountSmoother
from qonaq.zones.models import ZoneSet


@dataclass
class _Stay:
    first_ts: float
    last_ts: float
    sid: int = 0  # 0 — ещё не подтверждён
    alerted: bool = False
    preexisting: bool = False


@dataclass
class _ZoneTracker:
    """Пребывания треков в одной зоне с подтверждением входа и выхода."""

    zone_id: str
    enter_min_s: float
    exit_min_s: float
    stays: dict[int, _Stay] = field(default_factory=dict)

    def update(self, ts: float, ids: set[int], first_ts: float, grace_s: float):
        """Возвращает (подтверждённые входы, завершённые пребывания)."""
        started, ended = [], []
        for tid in ids:
            st = self.stays.get(tid)
            if st is None:
                st = self.stays[tid] = _Stay(ts, ts, preexisting=ts - first_ts <= grace_s)
            st.last_ts = ts
        for tid, st in list(self.stays.items()):
            if st.sid == 0 and st.last_ts - st.first_ts >= self.enter_min_s:
                started.append((tid, st))
            if ts - st.last_ts >= self.exit_min_s:
                del self.stays[tid]
                if st.sid:
                    ended.append((tid, st))
        return started, ended

    def confirmed(self) -> list[_Stay]:
        return [s for s in self.stays.values() if s.sid]


class ClinicEngine:
    profile = "clinic"
    GRACE_S = 5.0

    def __init__(self, zones: ZoneSet, site: SiteConfig | None = None):
        site = site or SiteConfig()
        self.th: ClinicThresholds = site.clinic
        self.al: AlertThresholds = site.alerts
        self.zones = zones
        th = self.th
        self.waiting = {z.id: _ZoneTracker(z.id, th.enter_min_s, th.exit_min_s) for z in zones.by_type("waiting_area")}
        self.desks = {z.id: _ZoneTracker(z.id, th.service_min_s, th.exit_min_s) for z in zones.by_type("service_point")}
        self._desk_has_staff_area = {z.id: bool(z.staff_polygon) for z in zones.by_type("service_point")}
        w = (th.dropout_tolerance_s, th.presence_on_ratio, th.presence_off_ratio)
        self._queue = {zid: PresenceFilter(*w) for zid in self.desks}
        self._staff = {zid: PresenceFilter(*w) for zid in self.desks}
        self._count = {zid: CountSmoother() for zid in list(self.waiting) + list(self.desks)}
        self._unattended_alerted: dict[str, bool] = dict.fromkeys(self.desks, False)
        self._alert_seq = 0
        self._desk_seen: dict[int, float] = {}  # id -> когда последний раз был у любой стойки
        self._sid = 0
        self._first_ts: float | None = None
        self._ts = 0.0

    def _next_sid(self) -> int:
        self._sid += 1
        return self._sid

    def update(self, observations: list[ZoneObservation]) -> list[Event]:
        out: list[Event] = []
        if not observations:
            return out
        ts = max(o.ts for o in observations)
        self._ts = ts
        if self._first_ts is None:
            self._first_ts = ts
        by_zone = {o.zone_id: o for o in observations}
        ids_of = lambda o: set(o.guest_tracks) | set(o.unknown_tracks) | set(o.staff_tracks)  # noqa: E731

        # --- стойки: сначала они, чтобы знать, кто дошёл до стойки
        for zid, tracker in self.desks.items():
            o = by_zone.get(zid)
            if o is None:
                continue
            ids = ids_of(o)
            for tid in ids:
                self._desk_seen[tid] = ts
            self._count[zid].update(ts, len(ids))
            started, ended = tracker.update(ts, ids, self._first_ts, self.GRACE_S)
            for _, st in started:
                st.sid = self._next_sid()
                out.append(Event(st.first_ts, zid, "service_start", {"sid": st.sid, "detected_ts": ts}))
            for _, st in ended:
                out.append(Event(st.last_ts, zid, "service_end", {
                    "sid": st.sid, "duration_s": round(st.last_ts - st.first_ts, 1), "outcome": "completed",
                    "detected_ts": ts}))
            # очередь есть, сотрудника нет
            self._queue[zid].update(ts, bool(ids))
            if self._desk_has_staff_area[zid] and o.staff_behind_desk is not None:
                staff = self._staff[zid]
                staff.update(ts, bool(o.staff_behind_desk))
                if staff.state:
                    self._unattended_alerted[zid] = False
                elif self._queue[zid].state and not self._unattended_alerted[zid]:
                    since = max(staff.since or ts, self._queue[zid].since or ts)
                    if ts - since >= self.al.desk_unattended_s:
                        self._unattended_alerted[zid] = True
                        self._alert_seq += 1
                        out.append(Event(ts, zid, "alert", {"kind": "desk_unattended", "seq": self._alert_seq,
                                                            "unattended_s": round(ts - since, 1)}))

        # --- зоны ожидания
        for zid, tracker in self.waiting.items():
            o = by_zone.get(zid)
            if o is None:
                continue
            ids = ids_of(o)
            self._count[zid].update(ts, len(ids))
            started, ended = tracker.update(ts, ids, self._first_ts, self.GRACE_S)
            for _, st in started:
                st.sid = self._next_sid()
                out.append(Event(st.first_ts, zid, "wait_start", {"sid": st.sid, "preexisting": st.preexisting,
                                                                  "detected_ts": ts}))
            for tid, st in ended:
                seen = self._desk_seen.get(tid)
                served = seen is not None and seen >= st.first_ts and seen - st.last_ts <= self.th.handoff_s
                out.append(Event(st.last_ts, zid, "wait_end", {
                    "sid": st.sid, "duration_s": round(st.last_ts - st.first_ts, 1),
                    "outcome": "served" if served else "left", "preexisting": st.preexisting, "detected_ts": ts}))
            for st in tracker.confirmed():
                if not st.alerted and ts - st.first_ts >= self.al.long_wait_s:
                    st.alerted = True
                    out.append(Event(ts, zid, "alert", {"kind": "long_wait", "sid": st.sid,
                                                        "wait_s": round(ts - st.first_ts, 1)}))
        for tid in [t for t, seen in self._desk_seen.items() if ts - seen > 600]:
            del self._desk_seen[tid]
        return out

    def snapshot(self, ts: float) -> dict[str, dict[str, Any]]:
        snap: dict[str, dict[str, Any]] = {}
        for zid, tr in self.waiting.items():
            stays = tr.confirmed()
            longest = max((ts - s.first_ts for s in stays), default=0.0)
            snap[zid] = {"zone_id": zid, "type": "waiting_area", "state": "WAITING" if stays else "EMPTY",
                         "count": len(stays), "longest_s": round(longest, 1),
                         "alert": "long_wait" if longest >= self.al.long_wait_s else None}
        for zid in self.desks:
            staffed = self._staff[zid].state if self._desk_has_staff_area[zid] else None
            n = self._count[zid].value
            state = "EMPTY" if not n else ("UNATTENDED" if staffed is False else "ATTENDED")
            snap[zid] = {"zone_id": zid, "type": "service_point", "state": state, "count": n, "staffed": staffed,
                         "alert": "desk_unattended" if self._unattended_alerted[zid] else None}
        return snap

    def summary(self, ts: float) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "waiting": sum(len(t.confirmed()) for t in self.waiting.values()),
            "at_desk": sum(self._count[z].value for z in self.desks),
            "tables": 0, "occupied": 0, "dirty": 0, "free": 0, "load_pct": 0, "queue": 0,
        }
