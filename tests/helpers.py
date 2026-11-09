"""Синтетические таймлайны для тестов логики: ни одного кадра видео."""
from __future__ import annotations

from qonaq.logic.base import Event, ZoneObservation


class Timeline:
    """Кормит машину(ы) наблюдениями с шагом 1/fps и копит события."""

    def __init__(self, machine, fps: float = 5.0, zone_id: str | None = None):
        self.m = machine
        self.dt = 1.0 / fps
        self.t = 0.0
        self.zone_id = zone_id or getattr(machine, "zone_id", "T01")
        self.events: list[Event] = []

    def run(self, seconds: float, guests=(), staff=(), unknown=(), flicker: int = 0, **kw) -> Timeline:
        """flicker=N: каждое N-е наблюдение детекция «пропадает» (пустой кадр)."""
        n = int(round(seconds / self.dt))
        for i in range(n):
            drop = flicker and (i % flicker == flicker - 1)
            obs = ZoneObservation(
                ts=round(self.t, 3),
                zone_id=self.zone_id,
                guest_tracks=() if drop else tuple(guests),
                staff_tracks=() if drop else tuple(staff),
                unknown_tracks=() if drop else tuple(unknown),
                **kw,
            )
            self.events.extend(self.m.update(obs))
            self.t += self.dt
        return self

    def types(self, *skip: str) -> list[str]:
        skip = skip or ("state_change",)
        return [e.type for e in self.events if e.type not in skip]

    def of(self, type_: str) -> list[Event]:
        return [e for e in self.events if e.type == type_]

    def alerts(self) -> list[str]:
        return [e.payload["kind"] for e in self.events if e.type == "alert"]
