"""Профиль `billiard` (SPRINT3 B2): стол «в игре» по движению шаров на сукне, а не по людям рядом.

Чистая логика от ZoneObservation.felt (FeltSignal) — видео не знает. Люди ходят между столами, играют на
одном и стоят у другого, поэтому присутствие людей у стола в решении не участвует вообще.

Состояния: IDLE (свободен) → IN_PLAY (идёт игра) ⇄ PAUSED (эпизодов нет idle_s, сессия ещё не закрыта) → IDLE.

- Эпизод движения — подряд идущие кадры с felt.motion (разрыв ≤ episode_gap_s); засчитывается, если длится
  ≥ episode_min_s и в нём ≥ episode_min_frames кадров (одиночный «блик» — не эпизод).
- Игра началась, когда за start_window_s набралось ≥ start_episodes эпизодов; начало сессии — задним числом,
  с первого из них. Один удар «между делом» (игрок соседнего стола толкнул шар) сессию не открывает.
- Сессия кончилась: нет эпизодов pause_max_s (паузы короче не рвут сессию), или лампа над столом погасла,
  или стол накрыт. Конец — конец последнего эпизода (консервативно: «сыграно» не больше, чем видно движение).
- Кадры с global_change (свет мигнул) и motion=None — нейтральны: ни движение, ни его отсутствие.
- Обрыв камеры (> blind_gap_s без наблюдений) «замораживает» время, как в ресторане (D-035): пауза
  не растёт, пока мы ничего не видим.

События: game_started, game_ended (payload: start, end, duration_s, episodes, pauses, reason), state_change,
lamp_on / lamp_off, table_covered / table_uncovered.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from qonaq.config import BilliardThresholds, SiteConfig
from qonaq.logic.base import Event, PresenceFilter, ZoneObservation, fmt_mmss
from qonaq.zones.models import ZoneSet


class BState(StrEnum):
    IDLE = "IDLE"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"


@dataclass
class _Episode:
    start: float
    end: float
    frames: int = 1


class BilliardTableMachine:
    def __init__(self, zone_id: str, th: BilliardThresholds | None = None):
        self.zone_id = zone_id
        self.th = th or BilliardThresholds()
        self.state = BState.IDLE
        self.state_since = 0.0
        self._cur: _Episode | None = None  # открытый эпизод
        self._recent: deque[_Episode] = deque()  # засчитанные эпизоды за start_window_s (для старта)
        self._session: dict[str, Any] | None = None
        self._last_ts: float | None = None
        self._blind_s = 0.0
        self.lamp = PresenceFilter(self.th.lamp_window_s, 0.7, 0.3)
        self.covered = PresenceFilter(self.th.covered_window_s, 0.8, 0.2)
        self._lamp_known = False

    # ------------------------------------------------------------------ helpers
    def _set_state(self, new: BState, ts: float, out: list[Event], **payload: Any) -> None:
        if new == self.state:
            return
        out.append(Event(ts, self.zone_id, "state_change", {"from": self.state.value, "to": new.value, **payload}))
        self.state = new
        self.state_since = ts

    def _freeze(self, a: float, b: float) -> None:
        """Обрыв камеры с a по b: все отметки сдвигаются на длину обрыва (время вслепую не считается)."""
        d = b - a
        self._blind_s += d
        if self._cur is not None:
            self._cur.start += d
            self._cur.end += d
        for e in self._recent:
            e.start += d
            e.end += d
        if self._session is not None:
            self._session["last_end"] += d
            self._session["blind_s"] += d
        self.lamp.on_gap(a, b)
        self.covered.on_gap(a, b)

    def _close_episode(self, out: list[Event]) -> None:
        ep, self._cur = self._cur, None
        if ep is None:
            return
        th = self.th
        if ep.end - ep.start < th.episode_min_s or ep.frames < th.episode_min_frames:
            return
        self._recent.append(ep)
        if self._session is not None:
            s = self._session
            gap = ep.start - s["last_end"]
            if gap >= th.idle_s:
                s["pauses"].append(round(gap, 1))
            s["last_end"] = max(s["last_end"], ep.end)
            s["episodes"] += 1
            if self.state == BState.PAUSED:
                self._set_state(BState.IN_PLAY, ep.start, out)
            return
        while self._recent and self._recent[0].end < ep.end - th.start_window_s:
            self._recent.popleft()
        if len(self._recent) >= th.start_episodes:
            first = self._recent[0]
            self._session = {"start": first.start, "last_end": ep.end, "episodes": len(self._recent),
                             "pauses": [], "blind_s": 0.0, "lamp_off_seen": False}
            out.append(Event(first.start, self.zone_id, "game_started",
                             {"start": round(first.start, 3), "detected_ts": round(ep.end, 3),
                              "episodes": len(self._recent)}))
            self._set_state(BState.IN_PLAY, first.start, out)
            self._recent.clear()

    def _end_session(self, ts: float, reason: str, out: list[Event]) -> None:
        s, self._session = self._session, None
        if s is None:
            return
        end = s["last_end"]
        out.append(Event(end, self.zone_id, "game_ended", {
            "start": round(s["start"], 3), "end": round(end, 3), "duration_s": round(end - s["start"], 1),
            "episodes": s["episodes"], "pauses": s["pauses"], "blind_s": round(s["blind_s"], 1),
            "reason": reason, "detected_ts": round(ts, 3)}))
        self._set_state(BState.IDLE, end, out, reason=reason)
        self._recent.clear()

    # ------------------------------------------------------------------ main
    def update(self, obs: ZoneObservation) -> list[Event]:
        out: list[Event] = []
        ts = obs.ts
        th = self.th
        if self._last_ts is not None and ts - self._last_ts > th.blind_gap_s:
            self._freeze(self._last_ts, ts)
        if self._last_ts is None:
            self.state_since = ts
        self._last_ts = ts
        f = obs.felt

        # лампа и «стол накрыт» — с гистерезисом
        if f is not None and f.lamp_on is not None and not f.global_change:
            self._lamp_known = True
            if self.lamp.update(ts, f.lamp_on):
                out.append(Event(self.lamp.since or ts, self.zone_id, "lamp_on" if self.lamp.state else "lamp_off"))
                if not self.lamp.state and self._session is not None:
                    self._session["lamp_off_seen"] = True
                    if th.end_on_lamp_off:
                        self._close_episode(out)
                        self._end_session(ts, "lamp_off", out)
        if f is not None and f.covered is not None:
            if self.covered.update(ts, f.covered):
                out.append(Event(self.covered.since or ts, self.zone_id,
                                 "table_covered" if self.covered.state else "table_uncovered"))
                if self.covered.state and self._session is not None and th.end_on_covered:
                    self._close_episode(out)
                    self._end_session(ts, "covered", out)

        # эпизоды движения
        usable = f is not None and f.motion is not None and not f.global_change and not self.covered.state
        if self._cur is not None and ts - self._cur.end > th.episode_gap_s:
            self._close_episode(out)
        if usable and f.motion:
            if self._cur is None:
                self._cur = _Episode(ts, ts, 1)
            else:
                self._cur.end = ts
                self._cur.frames += 1

        # пауза и конец сессии
        if self._session is not None:
            last = self._session["last_end"] if self._cur is None else ts
            quiet = ts - last
            if quiet >= th.pause_max_s:
                self._end_session(ts, "pause", out)
            elif quiet >= th.idle_s and self.state == BState.IN_PLAY:
                self._set_state(BState.PAUSED, ts, out)
        return out

    def finish(self, ts: float) -> list[Event]:
        """Конец записи / штатная остановка: открытая сессия закрывается по последнему эпизоду."""
        out: list[Event] = []
        self._close_episode(out)
        self._end_session(ts, "shutdown", out)
        return out

    def snapshot(self, ts: float) -> dict[str, Any]:
        s = self._session
        sess = (max(s["last_end"], ts if self._cur else s["last_end"]) - s["start"]) if s else 0.0
        label = {BState.IDLE: "свободен", BState.IN_PLAY: f"игра {fmt_mmss(sess)}",
                 BState.PAUSED: f"пауза, игра {fmt_mmss(sess)}"}[self.state]
        return {"zone_id": self.zone_id, "type": "billiard_table", "state": self.state.value,
                "since": round(self.state_since, 1), "session_s": round(sess, 1),
                "session_start": round(s["start"], 1) if s else None,
                "lamp_on": self.lamp.state if self._lamp_known else None, "covered": self.covered.state,
                "label": label, "alert": None}


class BilliardEngine:
    """Бильярдные столы (billiard_table) + места у барной стойки (table → обычный ресторанный автомат)."""

    profile = "billiard"

    def __init__(self, zones: ZoneSet, site: SiteConfig | None = None, pos=None):
        from qonaq.logic.engine import RestaurantEngine

        site = site or SiteConfig(profile="billiard")
        self.site = site
        self.zones = zones
        self.tables = {z.id: BilliardTableMachine(z.id, site.billiard) for z in zones.by_type("billiard_table")}
        self.bar = RestaurantEngine(zones, site, pos=pos)  # зоны table/entrance — как в ресторане

    def update(self, observations: list[ZoneObservation]) -> list[Event]:
        out: list[Event] = []
        rest = []
        for obs in observations:
            m = self.tables.get(obs.zone_id)
            if m is not None:
                out.extend(m.update(obs))
            else:
                rest.append(obs)
        if rest:
            out.extend(self.bar.update(rest))
        return out

    def finish(self, ts: float) -> list[Event]:
        out: list[Event] = []
        for m in self.tables.values():
            out.extend(m.finish(ts))
        return out

    def snapshot(self, ts: float) -> dict[str, dict[str, Any]]:
        snap = self.bar.snapshot(ts)
        snap.update({zid: m.snapshot(ts) for zid, m in self.tables.items()})
        return snap

    def summary(self, ts: float) -> dict[str, Any]:
        st = [m.state for m in self.tables.values()]
        bar = self.bar.summary(ts)
        return {"profile": self.profile, "billiard_tables": len(st),
                "in_play": sum(1 for s in st if s == BState.IN_PLAY),
                "paused": sum(1 for s in st if s == BState.PAUSED),
                "idle": sum(1 for s in st if s == BState.IDLE),
                "lamp_on": sum(1 for m in self.tables.values() if m.lamp.state),
                "bar_seats": bar.get("tables", 0), "bar_occupied": bar.get("occupied", 0),
                "tables": len(st), "occupied": sum(1 for s in st if s != BState.IDLE),
                "load_pct": round(100 * sum(1 for s in st if s != BState.IDLE) / len(st)) if st else 0}
