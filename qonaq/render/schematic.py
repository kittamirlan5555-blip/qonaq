"""Схематичный демо-ролик ресторана (B6): план зала сверху, фигуры-схемы, настоящая логика и панель.

Без чужих кадров и без датасетов — ролик можно показывать и публиковать (плашка «демонстрация на тестовых
данных» остаётся). Честно о том, что внутри: распознавание людей здесь **не участвует** — кто где стоит, берётся
из сценария; дальше всё как в работе: те же наблюдения «кто в зоне стола» → RestaurantEngine → события, алерт,
та же панель. Это показывает, что видит администратор, а не качество детектора (оно — в EVAL).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qonaq.config import SiteConfig
from qonaq.logic.base import ZoneObservation
from qonaq.zones.models import Zone, ZoneSet

PX = 120.0  # пикселей на метр
SIZE = (960, 540)


@dataclass
class Person:
    id: int
    role: str  # guest | staff
    path: list[tuple[float, float, float]]  # (t, x, y) в метрах; между точками — линейно; вне [t0, tN] — нет
    color: tuple[int, int, int]


TABLES = {"T1": (1.6, 1.4), "T2": (4.0, 1.4), "T3": (6.4, 1.4), "T4": (2.8, 3.4)}  # центры, стол 0,9 × 0,9 м
SEAT = 0.62  # от центра стола до сидящего


def seat(tid: str, k: int) -> tuple[float, float]:
    cx, cy = TABLES[tid]
    return [(cx - SEAT, cy), (cx + SEAT, cy), (cx, cy - SEAT), (cx, cy + SEAT)][k]


def edge(tid: str) -> tuple[float, float]:
    """Где стоит официант у стола (со стороны прохода)."""
    cx, cy = TABLES[tid]
    return cx + 0.25, cy + 0.75


DOOR = (7.7, 4.2)
KITCHEN = (0.3, 4.2)


def scenario() -> tuple[list[Person], dict[str, list[tuple[float, float]]]]:
    """Сюжет на ~14 минут сцены. Стол 1 — главная история: сели → 3 минуты без официанта → алерт → подошёл →
    ушли → стол не убран → убрали. Стол 2 — гости уже сидят, Стол 3 — мимо проходят, Стол 4 — быстрый подход."""
    G = [(70, 110, 170), (60, 140, 90), (150, 90, 60), (120, 70, 140), (40, 120, 200), (90, 90, 90)]
    staff = (35, 35, 35)
    P: list[Person] = []

    def walk_in(t: float, tid: str, k: int, leave: float | None) -> list[tuple[float, float, float]]:
        s = seat(tid, k)
        path = [(t, *DOOR), (t + 12, s[0], s[1] + 0.05)]
        if leave is not None:
            path += [(leave, *s), (leave + 14, *DOOR)]
        else:
            path += [(10_000, *s)]
        return path

    # Стол 1: двое заходят на 20-й секунде, уходят на 640-й
    P.append(Person(1, "guest", walk_in(20, "T1", 0, 640), G[0]))
    P.append(Person(2, "guest", walk_in(22, "T1", 1, 641), G[1]))
    # Стол 2: сидят с самого начала, уходят на 900-й
    P.append(Person(3, "guest", [(0, *seat("T2", 0)), (900, *seat("T2", 0)), (914, *DOOR)], G[2]))
    P.append(Person(4, "guest", [(0, *seat("T2", 1)), (901, *seat("T2", 1)), (915, *DOOR)], G[3]))
    # Стол 4: садятся на 300-й, официант через минуту
    P.append(Person(5, "guest", walk_in(300, "T4", 2, None), G[4]))
    # прохожий мимо Стола 3
    P.append(Person(6, "guest", [(120, *DOOR), (135, 6.4, 2.25), (150, 4.0, 2.4), (175, *KITCHEN)], G[5]))
    # официант: в зале ходит, к Столу 1 подходит только на 262-й (ожидание ~3:50) — алерт на 3:00
    w = [(0, *KITCHEN), (30, 4.4, 2.6), (60, *edge("T2")), (75, *edge("T2")), (95, *KITCHEN),
         (360, *KITCHEN), (372, *edge("T4")), (384, *edge("T4")), (400, *KITCHEN)]
    w2 = [(240, *KITCHEN), (258, *edge("T1")), (274, *edge("T1")), (290, *KITCHEN),
          (470, *KITCHEN), (486, *edge("T1")), (496, *edge("T1")), (512, *KITCHEN),
          # уборка Стола 1 после ухода гостей
          (790, *KITCHEN), (805, *edge("T1")), (825, *edge("T1")), (840, *KITCHEN)]
    P.append(Person(10, "staff", w, staff))
    P.append(Person(11, "staff", w2, staff))
    dishes = {"T1": [(655, 825)], "T2": [(915, 10_000)]}
    return P, dishes


def pos(p: Person, t: float) -> tuple[float, float] | None:
    if t < p.path[0][0] or t > p.path[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(p.path, p.path[1:], strict=False):
        if t0 <= t <= t1:
            k = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + k * (x1 - x0), y0 + k * (y1 - y0)
    return p.path[-1][1], p.path[-1][2]


def moving(p: Person, t: float) -> bool:
    a, b = pos(p, t), pos(p, t + 1.0)
    return a is not None and b is not None and math.hypot(a[0] - b[0], a[1] - b[1]) > 0.05


def zones() -> ZoneSet:
    zs = []
    for tid, (cx, cy) in TABLES.items():
        r = 0.95
        zs.append(Zone(id=tid, type="table", name=f"Стол {tid[1:]}",
                       polygon=[((cx - r) * PX, (cy - r) * PX), ((cx + r) * PX, (cy - r) * PX),
                                ((cx + r) * PX, (cy + r) * PX), ((cx - r) * PX, (cy + r) * PX)]))
    return ZoneSet(camera_id="schematic", frame_size=SIZE, zones=zs)


def render(t: float, people: list[Person], dishes: dict[str, list[tuple[float, float]]]) -> np.ndarray:
    img = np.full((SIZE[1], SIZE[0], 3), (52, 56, 62), np.uint8)
    for x in range(0, SIZE[0], 48):
        cv2.line(img, (x, 0), (x, SIZE[1]), (58, 62, 68), 1)
    cv2.rectangle(img, (0, int(3.9 * PX)), (int(0.7 * PX), SIZE[1]), (70, 80, 95), -1)
    cv2.putText(img, "kitchen", (6, SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.rectangle(img, (int(7.4 * PX), int(4.35 * PX)), (SIZE[0], SIZE[1]), (90, 110, 90), -1)
    for tid, (cx, cy) in TABLES.items():
        for k in range(4):
            sx, sy = seat(tid, k)
            cv2.circle(img, (int(sx * PX), int(sy * PX)), 13, (80, 95, 110), -1)  # стулья
        p0 = (int((cx - 0.45) * PX), int((cy - 0.45) * PX))
        p1 = (int((cx + 0.45) * PX), int((cy + 0.45) * PX))
        cv2.rectangle(img, p0, p1, (150, 175, 195), -1)
        cv2.rectangle(img, p0, p1, (110, 130, 150), 2)
        if any(a <= t < b for a, b in dishes.get(tid, [])):
            for dx, dy in ((-0.2, -0.15), (0.18, 0.1), (0.0, 0.22)):
                cv2.circle(img, (int((cx + dx) * PX), int((cy + dy) * PX)), 9, (240, 240, 240), -1)
                cv2.circle(img, (int((cx + dx) * PX), int((cy + dy) * PX)), 9, (160, 160, 160), 1)
    for p in people:
        q = pos(p, t)
        if q is None:
            continue
        c = (int(q[0] * PX), int(q[1] * PX))
        cv2.ellipse(img, (c[0] + 5, c[1] + 6), (24, 17), 0, 0, 360, (40, 42, 46), -1)
        cv2.ellipse(img, c, (24, 15), 0, 0, 360, p.color, -1)
        if p.role == "staff":
            cv2.ellipse(img, c, (24, 15), 0, 0, 360, (250, 250, 250), 2)  # белая окантовка — форма
        cv2.circle(img, c, 9, (60, 80, 110), -1)
    return img


def observations(t: float, people: list[Person], zs: ZoneSet) -> list[ZoneObservation]:
    out = []
    for z in zs.zones:
        x0, y0, x1, y1 = z.bbox()
        g, s, u, standing = [], [], [], []
        for p in people:
            q = pos(p, t)
            if q is None or not (x0 <= q[0] * PX <= x1 and y0 <= q[1] * PX <= y1):
                continue
            if p.role == "staff":
                s.append(p.id)
            elif moving(p, t):
                u.append(p.id)  # идущий гость — ещё не «гость за столом» (как роли в пайплайне: сначала сел)
            else:
                g.append(p.id)
            if p.role == "staff" or moving(p, t):
                standing.append(p.id)
        out.append(ZoneObservation(ts=t, zone_id=z.id, guest_tracks=tuple(g), staff_tracks=tuple(s),
                                   unknown_tracks=tuple(u), standing_tracks=tuple(standing),
                                   guest_confidence=0.9 if g else None))
    return out


def run(out_dir: Path, fps: float = 2.0, duration: float = 960.0) -> Path:
    """Сцена → журнал events.sqlite + demo.mp4 (кадр + панель, как в работе)."""
    from qonaq.events.store import EventStore
    from qonaq.logic.engine import RestaurantEngine
    from qonaq.pipeline import FrameResult, PersonView
    from qonaq.render.annotate import DemoRenderer
    from qonaq.render.sink import VideoSink

    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "events.sqlite"
    if db.exists():
        db.unlink()
    store = EventStore(db)
    site = SiteConfig()
    site.thresholds.preexisting_grace_s = 5.0
    zs = zones()
    eng = RestaurantEngine(zs, site)
    people, dishes = scenario()
    demo = DemoRenderer(zs, site, blur=False)
    sink = VideoSink(out_dir / "demo.mp4", fps)
    for i in range(int(duration * fps)):
        t = i / fps
        evs = eng.update(observations(t, people, zs))
        for e in evs:
            e.camera_id = "schematic"
        store.add(evs)
        frame = render(t, people, dishes)
        pv = []
        for p in people:
            q = pos(p, t)
            if q is None:
                continue
            x, y = q[0] * PX, q[1] * PX
            zn = [z.id for z in zs.zones if z.bbox()[0] <= x <= z.bbox()[2] and z.bbox()[1] <= y <= z.bbox()[3]]
            pv.append(PersonView(p.id, (x - 26, y - 20, x + 26, y + 20), p.role, 0.9, zn))
        summary = eng.summary(t)
        summary["people"] = len(pv)
        sink.write(demo.render(FrameResult("schematic", t, frame, pv, [], [], eng.snapshot(t), summary, evs, i)))
    sink.close()
    store.set_meta(time_base="video", site_id="schematic", profile="restaurant", scene_s=duration, tables=len(zs.zones))
    store.close()
    return out_dir


def cmd_schematic_reel(args) -> int:
    from qonaq.render.reel import build_reel

    run_dir = Path("runs") / "schematic_reel"
    if args.rerun or not (run_dir / "demo.mp4").exists():
        run(run_dir)
    out = Path(args.out if args.out != "runs/demo_reel.mp4" else "runs/demo_reel_schematic.mp4")
    build_reel(run_dir, None, out, scene_fps=2.0,
               intro_sub="Схема зала сверху; кто где — из сценария, логика и панель — настоящие")
    print(f"Демо-ролик (схема): {out}")
    return 0
