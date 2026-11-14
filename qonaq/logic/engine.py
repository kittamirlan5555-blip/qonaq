"""Движки профилей: набор зон + профиль логики -> события. Общий интерфейс для restaurant и clinic."""
from __future__ import annotations

from collections import deque
from typing import Any, Protocol

from qonaq.config import SiteConfig
from qonaq.logic.base import Event, ZoneObservation
from qonaq.logic.restaurant import OCCUPIED, TableMachine
from qonaq.zones.models import ZoneSet


class Engine(Protocol):
    profile: str

    def update(self, observations: list[ZoneObservation]) -> list[Event]: ...
    def snapshot(self, ts: float) -> dict[str, dict[str, Any]]: ...
    def summary(self, ts: float) -> dict[str, Any]: ...


class CountSmoother:
    """Число людей в зоне, сглаженное медианой за окно (очередь у входа и т.п.)."""

    def __init__(self, window_s: float = 3.0):
        self.window_s = window_s
        self._obs: deque[tuple[float, int]] = deque()
        self.value = 0

    def update(self, ts: float, n: int) -> bool:
        self._obs.append((ts, n))
        while self._obs and self._obs[0][0] < ts - self.window_s:
            self._obs.popleft()
        vals = sorted(v for _, v in self._obs)
        new = vals[len(vals) // 2]
        changed = new != self.value
        self.value = new
        return changed


class RestaurantEngine:
    profile = "restaurant"

    def __init__(self, zones: ZoneSet, site: SiteConfig | None = None, pos=None):
        site = site or SiteConfig()
        self.zones = zones
        self.tables = {
            z.id: TableMachine(z.id, site.thresholds, site.alerts, pos=pos) for z in zones.by_type("table")
        }
        self.entrances = {z.id: CountSmoother() for z in zones.by_type("entrance")}
        self._ts = 0.0

    def update(self, observations: list[ZoneObservation]) -> list[Event]:
        out: list[Event] = []
        for obs in observations:
            self._ts = max(self._ts, obs.ts)
            m = self.tables.get(obs.zone_id)
            if m is not None:
                out.extend(m.update(obs))
                continue
            q = self.entrances.get(obs.zone_id)
            if q is not None:
                n = len(set(obs.guest_tracks) | set(obs.unknown_tracks))
                if q.update(obs.ts, n):
                    out.append(Event(obs.ts, obs.zone_id, "queue", {"count": q.value}))
        return out

    def snapshot(self, ts: float) -> dict[str, dict[str, Any]]:
        snap = {zid: m.snapshot(ts) for zid, m in self.tables.items()}
        for zid, q in self.entrances.items():
            snap[zid] = {"zone_id": zid, "type": "entrance", "state": "QUEUE" if q.value else "EMPTY",
                         "count": q.value, "alert": None}
        return snap

    def summary(self, ts: float) -> dict[str, Any]:
        states = [m.state for m in self.tables.values()]
        total = len(states)
        occupied = sum(1 for s in states if s in OCCUPIED)
        return {
            "profile": self.profile,
            "tables": total,
            "occupied": occupied,
            "waiting": sum(1 for s in states if s.value == "SEATED_WAITING"),
            "dirty": sum(1 for s in states if s.value == "NEEDS_CLEANING"),
            "free": sum(1 for s in states if s.value == "FREE"),
            "load_pct": round(100 * occupied / total) if total else 0,
            "queue": sum(q.value for q in self.entrances.values()),
        }


def create_engine(zones: ZoneSet, site: SiteConfig, pos=None) -> Engine:
    if site.profile == "clinic":
        from qonaq.logic.clinic import ClinicEngine

        return ClinicEngine(zones, site)
    if site.profile == "billiard":
        from qonaq.logic.billiard import BilliardEngine

        return BilliardEngine(zones, site, pos=pos)
    return RestaurantEngine(zones, site, pos=pos)
