"""Сценарий синтетического ролика и вывод «правды» (разметки) из него.

Разметка выводится из сценария по правилам SPEC §5 (не из кода логики!): сели → ждут → первый подход
персонала ≥ visit_min_s → обслуживают → гостей нет ≥ leave_min_s → ушли → визит персонала ≥ clean_min_s → убран.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Waypoint(BaseModel):
    t: float
    at: str | list[float]  # id зоны (центр полигона + смещение) или [x, y] — точка «ног»
    pose: Literal["sit", "stand"] = "stand"
    visible: bool = True
    seat: int = 0  # номер места у стола (смещение от центра), для нескольких гостей


class Actor(BaseModel):
    id: str
    kind: Literal["guest", "staff"]
    sprite: int = 0  # номер фигуры в банке (детерминированно)
    path: list[Waypoint]


class ZoneDef(BaseModel):
    id: str
    polygon: list[list[float]]
    seats: list[list[float]] = Field(default_factory=list)  # точки мест (ноги/центр сидящего)


class Background(BaseModel):
    video: str
    at_s: float = 0.0
    mode: Literal["frame", "median"] = "frame"  # median — медиана кадров (убирает идущих людей)


class Scenario(BaseModel):
    name: str
    description: str = ""
    background: Background
    fps: float = 2.0
    duration_s: float
    sit_h: float = 70.0  # высота сидящей фигуры, px (при y = scale_ref_y)
    stand_h: float = 120.0
    scale_ref_y: float | None = None  # перспектива: масштаб = 1 + scale_k * (y - scale_ref_y)
    scale_k: float = 0.0
    staff_venue: str = "C"  # из какого зала форма официанта (одна форма на ролик)
    guest_venues: list[str] = Field(default_factory=lambda: ["C", "B"])
    zones: list[ZoneDef]
    actors: list[Actor]
    # B1: шум матрицы, σ уровней яркости. Без него неподвижная синтетика — побитово одинаковые кадры, то есть
    # «зависший поток» для монитора здоровья. Реальная камера всегда шумит. Ролики до SPRINT3 — без шума (0).
    sensor_noise: float = 1.5
    dishes: bool = True  # после ухода гостей на стол «ставится посуда» до уборки
    alerts_forbidden: list[str] = Field(default_factory=list)
    alerts_expected: list[str] = Field(default_factory=list)
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- состояние актёра во времени
def actor_state(a: Actor, t: float) -> Waypoint | None:
    """Последняя пройденная точка (позиция между точками интерполируется в render)."""
    cur = None
    for w in a.path:
        if w.t <= t:
            cur = w
    return cur


def presence_intervals(sc: Scenario, zone_id: str, kind: str, pose: str | None = None) -> list[tuple[float, float]]:
    """Интервалы, когда хотя бы один актёр вида kind стоит в точке `at == zone_id` (и видим)."""
    pts: list[tuple[float, float]] = []
    for a in sc.actors:
        if a.kind != kind:
            continue
        for i, w in enumerate(a.path):
            end = a.path[i + 1].t if i + 1 < len(a.path) else sc.duration_s
            if w.visible and w.at == zone_id and (pose is None or w.pose == pose) and end > w.t:
                pts.append((w.t, end))
    pts.sort()
    merged: list[list[float]] = []
    for s, e in pts:
        if merged and s <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def derive_events(sc: Scenario, visit_min_s: float = 4.0, leave_min_s: float = 90.0,
                  clean_min_s: float = 10.0) -> list[dict]:
    """Правда по каждой зоне из таймлайна актёров (SPEC §5)."""
    out: list[dict] = []
    for z in sc.zones:
        guests = presence_intervals(sc, z.id, "guest")
        staff = [iv for iv in presence_intervals(sc, z.id, "staff") if iv[1] - iv[0] >= visit_min_s]
        # сессии: склеиваем отсутствия гостей короче leave_min_s
        sessions: list[list[float]] = []
        for s, e in guests:
            if sessions and s - sessions[-1][1] < leave_min_s:
                sessions[-1][1] = e
            else:
                sessions.append([s, e])
        for k, (s, e) in enumerate(sessions):
            out.append({"t": round(s, 1), "zone": z.id, "event": "seated"})
            vis = [iv for iv in staff if s <= iv[0] < e]
            for j, iv in enumerate(vis):
                out.append({"t": round(iv[0], 1), "zone": z.id, "event": "served" if j == 0 else "staff_visit"})
            ended = e + leave_min_s <= sc.duration_s
            if not ended:
                continue
            out.append({"t": round(e, 1), "zone": z.id, "event": "left" if vis else "walkout_before_service"})
            nxt = sessions[k + 1][0] if k + 1 < len(sessions) else sc.duration_s
            cleaners = [iv for iv in presence_intervals(sc, z.id, "staff")
                        if iv[1] > e and iv[1] - max(iv[0], e) >= clean_min_s and iv[0] < nxt]
            if cleaners:
                out.append({"t": round(cleaners[0][1], 1), "zone": z.id, "event": "cleaned"})
    return sorted(out, key=lambda x: (x["t"], x["zone"]))


def dish_intervals(sc: Scenario, leave_min_s: float = 90.0, clean_min_s: float = 10.0) -> dict[str, list[tuple]]:
    """Посуда на столе: от ухода последнего гостя сессии до конца уборки (или конца ролика)."""
    res: dict[str, list[tuple]] = {}
    ev = derive_events(sc, leave_min_s=0.01, clean_min_s=clean_min_s)  # уход «сразу», чтобы поставить посуду
    for z in sc.zones:
        zev = [e for e in ev if e["zone"] == z.id]
        cur = None
        for e in zev:
            if e["event"] in ("left", "walkout_before_service"):
                cur = e["t"]
            elif e["event"] == "cleaned" and cur is not None:
                res.setdefault(z.id, []).append((cur, e["t"]))
                cur = None
            elif e["event"] == "seated" and cur is not None:
                res.setdefault(z.id, []).append((cur, e["t"]))
                cur = None
        if cur is not None:
            res.setdefault(z.id, []).append((cur, sc.duration_s))
    return res
