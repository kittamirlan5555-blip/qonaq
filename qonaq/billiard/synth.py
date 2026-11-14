"""Синтетика бильярдной (B4): зал сверху/под углом, столы, шары с простой физикой, люди, лампы, мерцание.

Без датасетов и чужих кадров: всё рисуется процедурно, поэтому ролик можно показывать (B6) и в нём известна
«правда» — интервалы игры по каждому столу. Это проверка видео-слоя сукна и логики, а не «кино»: люди —
схематичные фигуры сверху (голова + плечи + тень), шары — кружки.

Сценарий — YAML (configs/synth_billiard/*.yaml):
    name, fps, duration_s, view: top|angled, flicker (амплитуда мерцания), room_light (0..1),
    tables: [{id, origin: [x, y] м — левый верхний угол сукна, size: [2.54, 1.27], felt: green|blue}],
    timeline:
      - {table: B1, game: [t0, t1], every: [25, 70], seed: 1}   # игра: удары через 25–70 с
      - {table: B2, shot: 900}                                  # одиночный удар
      - {table: B1, lamp: [[0, 3600]]}                          # лампа горит (по умолчанию — всегда)
      - {table: B1, cover: [3000, 3600]}                        # стол накрыт
      - {table: B2, brush: [1000, 1300]}                        # протирка сукна щёткой
      - {table: B2, lean: [1200, 1500]}                         # стоит, опершись на борт, не играет
      - {walk: [[t, x, y], ...]}                                # человек идёт по залу (м)
      - {lights_off: [t0, t1]}                                  # общий свет в зале выключен
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PX_M = 120.0  # пикселей на метр на плане пола
FLOOR = (8.0, 4.5)  # м
BALL_R = 0.0285
FELT_COLORS = {"green": (60, 120, 30), "blue": (150, 95, 25)}  # BGR
RAIL_W = 0.13
POCKET_R = 0.065


@dataclass
class Ball:
    x: float
    y: float
    color: tuple[int, int, int]
    vx: float = 0.0
    vy: float = 0.0
    on: bool = True


@dataclass
class Table:
    id: str
    origin: tuple[float, float]
    size: tuple[float, float] = (2.54, 1.27)
    felt: str = "green"
    balls: list[Ball] = field(default_factory=list)

    def rack(self, rng: np.random.Generator) -> None:
        w, h = self.size
        self.balls = [Ball(0.25 * w, 0.5 * h, (235, 235, 235))]
        cols = [(0, 200, 230), (200, 60, 30), (30, 30, 200), (120, 30, 120), (0, 120, 255), (30, 120, 30),
                (20, 20, 110), (15, 15, 15), (0, 220, 255), (220, 90, 40), (40, 40, 230), (150, 50, 150),
                (40, 150, 255), (50, 140, 50), (30, 30, 130)]
        k = 0
        for row in range(5):
            for j in range(row + 1):
                x = 0.72 * w + row * 2 * BALL_R * 0.95
                y = 0.5 * h + (j - row / 2) * 2 * BALL_R * 1.05
                self.balls.append(Ball(x + rng.normal(0, 0.0003), y + rng.normal(0, 0.0003), cols[k % len(cols)]))
                k += 1

    def felt_polygon_m(self) -> list[tuple[float, float]]:
        x, y = self.origin
        w, h = self.size
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


class Physics:
    """Шары: трение качения, отскок от бортов, упругое столкновение равных масс, лузы."""

    def __init__(self, decel: float = 0.35):  # шар катится 3–9 с, как в жизни
        self.decel = decel

    def shoot(self, t: Table, rng: np.random.Generator) -> None:
        live = [b for b in t.balls if b.on]
        if len(live) < 4:
            t.rack(rng)
            live = t.balls
        cue = live[0]
        target = live[int(rng.integers(1, len(live)))]
        ang = math.atan2(target.y - cue.y, target.x - cue.x) + rng.normal(0, 0.05)
        v = float(rng.uniform(1.2, 3.0))
        cue.vx, cue.vy = v * math.cos(ang), v * math.sin(ang)

    def moving(self, t: Table) -> bool:
        return any(b.on and (b.vx or b.vy) for b in t.balls)

    def step(self, t: Table, dt: float) -> None:
        w, h = t.size
        live = [b for b in t.balls if b.on]
        for b in live:
            sp = math.hypot(b.vx, b.vy)
            if sp == 0:
                continue
            ns = max(0.0, sp - self.decel * dt)
            if ns < 0.02:
                b.vx = b.vy = 0.0
                continue
            b.vx, b.vy = b.vx * ns / sp, b.vy * ns / sp
            b.x += b.vx * dt
            b.y += b.vy * dt
            if b.x < BALL_R or b.x > w - BALL_R:
                b.vx = -0.8 * b.vx
                b.x = min(max(b.x, BALL_R), w - BALL_R)
            if b.y < BALL_R or b.y > h - BALL_R:
                b.vy = -0.8 * b.vy
                b.y = min(max(b.y, BALL_R), h - BALL_R)
            for px, py in ((0, 0), (w / 2, 0), (w, 0), (0, h), (w / 2, h), (w, h)):
                if math.hypot(b.x - px, b.y - py) < POCKET_R and b is not live[0]:
                    b.on = False
                    b.vx = b.vy = 0.0
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                if 0 < d < 2 * BALL_R:
                    nx, ny = dx / d, dy / d
                    rel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny
                    if rel > 0:
                        a.vx -= rel * nx
                        a.vy -= rel * ny
                        b.vx += rel * nx
                        b.vy += rel * ny
                    ov = 2 * BALL_R - d
                    a.x -= nx * ov / 2
                    a.y -= ny * ov / 2
                    b.x += nx * ov / 2
                    b.y += ny * ov / 2


def _interp(path: list[list[float]], t: float) -> tuple[float, float] | None:
    if not path or t < path[0][0] or t > path[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:], strict=False):
        if t0 <= t <= t1:
            k = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + k * (x1 - x0), y0 + k * (y1 - y0)
    return path[-1][1], path[-1][2]


def _in(iv: list, t: float) -> bool:
    return any(a <= t < b for a, b in iv)


class Scene:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.fps = float(spec.get("fps", 2.0))
        self.duration = float(spec["duration_s"])
        self.view = spec.get("view", "angled")
        self.flicker = float(spec.get("flicker", 0.0))
        self.room_light = float(spec.get("room_light", 0.45))
        self.noise = float(spec.get("sensor_noise", 2.0))
        self.rng = np.random.default_rng(int(spec.get("seed", 0)))
        self.tables = {t["id"]: Table(t["id"], tuple(t["origin"]), tuple(t.get("size", (2.54, 1.27))),
                                      t.get("felt", "green")) for t in spec["tables"]}
        for t in self.tables.values():
            t.rack(self.rng)
        self.phys = Physics()
        self.lamp: dict[str, list] = {tid: [] for tid in self.tables}
        self.cover: dict[str, list] = {tid: [] for tid in self.tables}
        self.brush: dict[str, list] = {tid: [] for tid in self.tables}
        self.lean: dict[str, list] = {tid: [] for tid in self.tables}
        self.shots: dict[str, list[float]] = {tid: [] for tid in self.tables}
        self.games: dict[str, list[tuple[float, float]]] = {tid: [] for tid in self.tables}
        self.walks: list[list[list[float]]] = []
        self.lights_off: list = []
        has_lamp = set()
        for ev in spec.get("timeline", []):
            tid = ev.get("table")
            if "game" in ev:
                a, b = ev["game"]
                lo, hi = ev.get("every", [25, 70])
                r = np.random.default_rng(int(ev.get("seed", 1)))
                t = float(a)
                shots = []
                while t < b:
                    shots.append(t)
                    t += float(r.uniform(lo, hi))
                self.shots[tid] += shots
                self.games[tid].append((float(a), float(b)))
            if "shot" in ev:
                self.shots[tid].append(float(ev["shot"]))
            if "lamp" in ev:
                has_lamp.add(tid)
                self.lamp[tid] += [tuple(x) for x in ev["lamp"]]
            if "cover" in ev:
                self.cover[tid].append(tuple(ev["cover"]))
            if "brush" in ev:
                self.brush[tid].append(tuple(ev["brush"]))
            if "lean" in ev:
                self.lean[tid].append(tuple(ev["lean"]))
            if "walk" in ev:
                self.walks.append(ev["walk"])
            if "lights_off" in ev:
                self.lights_off.append(tuple(ev["lights_off"]))
        for tid in self.tables:
            if tid not in has_lamp:
                self.lamp[tid] = [(0.0, self.duration + 1)]
            self.shots[tid].sort()
        self.H = self._homography()
        self.size = (960, 540)
        self.ball_moving: dict[str, list[tuple[float, float]]] = {tid: [] for tid in self.tables}

    # ------------------------------------------------------------------ geometry
    def _homography(self) -> np.ndarray:
        W, H = FLOOR[0] * PX_M, FLOOR[1] * PX_M
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        if self.view == "top":
            dst = src
        else:  # камера под потолком у дальней стены, смотрит под углом: дальний край уже
            dst = np.float32([[150, 40], [W - 150, 40], [W, H], [0, H]])
        return cv2.getPerspectiveTransform(src, dst)

    def to_img(self, pts_m: list[tuple[float, float]]) -> list[list[float]]:
        p = np.float32([[x * PX_M, y * PX_M] for x, y in pts_m]).reshape(-1, 1, 2)
        return [[round(float(a), 1), round(float(b), 1)] for a, b in cv2.perspectiveTransform(p, self.H).reshape(-1, 2)]

    # ------------------------------------------------------------------ people
    def _players(self, tid: str, t: float) -> list[tuple[float, float, bool, tuple[float, float] | None]]:
        """Игроки стола: (x, y, наклонился, точка кия на сукне) в метрах."""
        tb = self.tables[tid]
        out = []
        if not any(a - 30 <= t < b + 60 for a, b in self.games[tid]):
            return out
        ox, oy = tb.origin
        w, h = tb.size
        shots = self.shots[tid]
        nxt = next((s for s in shots if s >= t), None)
        prev = next((s for s in reversed(shots) if s <= t), None)
        live = [b for b in tb.balls if b.on]
        cue = live[0] if live else Ball(w / 2, h / 2, (0, 0, 0))
        for k in range(2):
            mine = (shots.index(nxt) % 2 == k) if nxt is not None else k == 0
            leaning = mine and nxt is not None and nxt - t <= 4.0
            if leaning:
                # подходит к борту со стороны битка и наклоняется над сукном
                ang = math.atan2(cue.y - h / 2, cue.x - w / 2) + 0.4
                ex = min(max(cue.x + 0.45 * math.cos(ang), -0.25), w + 0.25)
                ey = min(max(cue.y + 0.45 * math.sin(ang), -0.25), h + 0.25)
                ex = ex if 0.05 < ex < w - 0.05 else ex
                if 0.1 < ex < w - 0.1 and 0.1 < ey < h - 0.1:  # проецируем к ближнему борту
                    dists = [ex, w - ex, ey, h - ey]
                    i = int(np.argmin(dists))
                    ex, ey = [(-0.05, ey), (w + 0.05, ey), (ex, -0.05), (ex, h + 0.05)][i]
                out.append((ox + ex, oy + ey, True, (ox + cue.x, oy + cue.y)))
            else:
                base = [(-0.55, h * 0.3), (w + 0.55, h * 0.7)][k]
                wob = 0.15 * math.sin(t / 7.0 + k)
                if prev is not None and t - prev < 6 and not mine:
                    base = (base[0], base[1] + 0.2)
                out.append((ox + base[0] + wob, oy + base[1], False, None))
        return out

    def _brusher(self, tid: str, t: float):
        tb = self.tables[tid]
        if not _in(self.brush[tid], t):
            return None
        ox, oy = tb.origin
        w, h = tb.size
        x = ox + (0.5 + 0.4 * math.sin(t / 3.0)) * w
        return (x, oy + h + 0.1, True, (x + 0.1 * math.sin(t), oy + h - 0.45 - 0.15 * math.sin(t / 2)))

    def _leaner(self, tid: str, t: float):
        tb = self.tables[tid]
        if not _in(self.lean[tid], t):
            return None
        ox, oy = tb.origin
        w, h = tb.size
        return (ox + 0.6 * w + 0.05 * math.sin(t / 5), oy - 0.12, True, None)

    # ------------------------------------------------------------------ render
    def render(self, t: float) -> np.ndarray:
        W, H = int(FLOOR[0] * PX_M), int(FLOOR[1] * PX_M)
        room = self.room_light * (0.25 if _in(self.lights_off, t) else 1.0)
        floor = np.zeros((H, W, 3), np.float32)
        floor[:] = (70, 75, 85)
        for i in range(0, W, 60):  # плитка пола
            floor[:, i:i + 2] *= 0.85
        floor *= room
        light = np.full((H, W), room, np.float32)
        for tid, tb in self.tables.items():
            on = _in(self.lamp[tid], t)
            x0, y0 = tb.origin
            w, h = tb.size
            r0 = (int((x0 - RAIL_W) * PX_M), int((y0 - RAIL_W) * PX_M))
            r1 = (int((x0 + w + RAIL_W) * PX_M), int((y0 + h + RAIL_W) * PX_M))
            if on:  # пятно света лампы
                light = np.maximum(light, self._spot(tid, W, H))
            cv2.rectangle(floor, r0, r1, (30, 50, 90), -1)  # борт (коричневый)
            f0 = (int(x0 * PX_M), int(y0 * PX_M))
            f1 = (int((x0 + w) * PX_M), int((y0 + h) * PX_M))
            if _in(self.cover[tid], t):
                cv2.rectangle(floor, (r0[0] - 3, r0[1] - 3), (r1[0] + 3, r1[1] + 3), (55, 55, 60), -1)
                continue
            cv2.rectangle(floor, f0, f1, FELT_COLORS.get(tb.felt, FELT_COLORS["green"]), -1)
            for px, py in ((0, 0), (w / 2, 0), (w, 0), (0, h), (w / 2, h), (w, h)):
                pc = (int((x0 + px) * PX_M), int((y0 + py) * PX_M))
                cv2.circle(floor, pc, int(POCKET_R * PX_M), (10, 10, 10), -1)
            for b in tb.balls:
                if b.on:
                    c = (int((x0 + b.x) * PX_M), int((y0 + b.y) * PX_M))
                    cv2.circle(floor, c, max(2, int(BALL_R * PX_M + 0.5)), b.color, -1, cv2.LINE_AA)
        img = floor * light[..., None]
        people = []
        for tid in self.tables:
            people += self._players(tid, t)
            for p in (self._brusher(tid, t), self._leaner(tid, t)):
                if p is not None:
                    people.append(p)
        for path in self.walks:
            p = _interp(path, t)
            if p is not None:
                people.append((p[0], p[1], False, None))
        for k, (x, y, lean, cue) in enumerate(people):
            self._draw_person(img, x, y, lean, cue, k)
        if self.flicker:
            img *= 1.0 + self.flicker * math.sin(2 * math.pi * 0.37 * t)
        img = np.clip(img, 0, 255).astype(np.uint8)
        out = cv2.warpPerspective(img, self.H, self.size, flags=cv2.INTER_LINEAR)
        if self.noise > 0:
            out = np.clip(out + self.rng.normal(0, self.noise, out.shape), 0, 255).astype(np.uint8)
        return out

    def _spot(self, tid: str, W: int, H: int) -> np.ndarray:
        if not hasattr(self, "_spots"):
            self._spots: dict[str, np.ndarray] = {}
        if tid not in self._spots:
            tb = self.tables[tid]
            x0, y0 = tb.origin
            w, h = tb.size
            cx, cy = int((x0 + w / 2) * PX_M), int((y0 + h / 2) * PX_M)
            spot = np.zeros((H, W), np.float32)
            cv2.ellipse(spot, (cx, cy), (int(0.75 * w * PX_M), int(0.95 * h * PX_M)), 0, 0, 360, 1.0, -1)
            self._spots[tid] = 0.35 + 0.65 * cv2.GaussianBlur(spot, (0, 0), 25)
        return self._spots[tid]

    def _draw_person(self, img, x, y, lean, cue, k) -> None:
        c = (int(x * PX_M), int(y * PX_M))
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, c[0] - 30), max(0, c[1] - 20), min(W, c[0] + 50), min(H, c[1] + 45)
        if x1 > x0 and y1 > y0:  # тень — полупрозрачный эллипс, только в своём прямоугольнике
            roi = img[y0:y1, x0:x1]
            m = np.zeros(roi.shape[:2], np.float32)
            cv2.ellipse(m, (c[0] + 10 - x0, c[1] + 12 - y0), (34, 26), 0, 0, 360, 0.35, -1)
            roi *= (1.0 - m)[..., None]
        body = [(40, 40, 45), (90, 60, 40), (50, 50, 110), (70, 90, 70)][k % 4]
        if cue is not None:
            cx, cy = int(cue[0] * PX_M), int(cue[1] * PX_M)
            cv2.line(img, c, (cx, cy), (120, 170, 200), 3, cv2.LINE_AA)  # кий / щётка
            if lean:  # рука от плеча к кисти — сплошная (как настоящая), кисть на 2/3 пути к точке кия
                mx, my = (c[0] + 2 * cx) // 3, (c[1] + 2 * cy) // 3
                cv2.line(img, c, (mx, my), body, 11, cv2.LINE_AA)
                cv2.circle(img, (mx, my), 8, (80, 120, 170), -1)
        cv2.ellipse(img, c, (30, 18), 0, 0, 360, body, -1)  # плечи
        cv2.circle(img, c, 11, (30, 40, 60), -1)  # голова

    # ------------------------------------------------------------------ time
    def frames(self):
        """(t, кадр) с шагом 1/fps; физика — 50 Гц."""
        dt = 0.02
        sim_t = 0.0
        pending = {tid: list(s) for tid, s in self.shots.items()}
        moving_since: dict[str, float | None] = {tid: None for tid in self.tables}
        n = int(self.duration * self.fps)
        for i in range(n):
            t = i / self.fps
            while sim_t < t:
                for tid, tb in self.tables.items():
                    was = self.phys.moving(tb)
                    while pending[tid] and pending[tid][0] <= sim_t:
                        pending[tid].pop(0)
                        if not _in(self.cover[tid], sim_t):
                            self.phys.shoot(tb, self.rng)
                    self.phys.step(tb, dt)
                    now = self.phys.moving(tb)
                    if now and not was:
                        moving_since[tid] = sim_t
                    if was and not now and moving_since[tid] is not None:
                        self.ball_moving[tid].append((moving_since[tid], sim_t))
                        moving_since[tid] = None
                sim_t += dt
            yield t, self.render(t)

    def truth(self) -> dict[str, Any]:
        """«Правда» по столам: интервалы игры [первый удар, конец движения шаров после последнего удара]."""
        out: dict[str, Any] = {}
        for tid in self.tables:
            sess = []
            for a, b in self.games[tid]:
                shots = [s for s in self.shots[tid] if a <= s < b]
                if not shots:
                    continue
                last = shots[-1]
                ends = [e for s, e in self.ball_moving[tid] if s >= last - 0.1]
                sess.append([round(shots[0], 1), round(ends[0] if ends else last + 3.0, 1)])
            out[tid] = {"games": sess, "single_shots": [s for s in self.shots[tid]
                                                        if not any(a <= s < b for a, b in self.games[tid])]}
        return out

    def zones(self) -> dict[str, Any]:
        return {"camera_id": f"synth_{self.spec['name']}", "frame_size": list(self.size),
                "zones": [{"id": tid, "type": "billiard_table", "polygon": self.to_img(tb.felt_polygon_m())}
                          for tid, tb in self.tables.items()]}


def build(scenario: Path, out_dir: Path, zones_dir: Path | None = None) -> dict[str, Any]:
    """Рендер сценария в mp4 + зоны + разметка (data/gt/<name>.yaml)."""
    from qonaq.render.sink import VideoSink

    spec = yaml.safe_load(Path(scenario).read_text(encoding="utf-8"))
    sc = Scene(spec)
    name = f"synth_{spec['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{name}.mp4"
    sink = VideoSink(video, sc.fps, crf=26)
    n = 0
    try:
        for _, frame in sc.frames():
            sink.write(frame)
            n += 1
    finally:
        sink.close()
    zdir = zones_dir or out_dir
    zdir.mkdir(parents=True, exist_ok=True)
    zpath = zdir / f"{name}.yaml"
    zpath.write_text(yaml.safe_dump(sc.zones(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    gt = {"video": video.name, "zones": str(zpath), "synthetic": True, "scenario": str(scenario),
          "fps": sc.fps, "duration_s": sc.duration, "tables": sc.truth()}
    gpath = out_dir / f"{name}.gt.yaml"
    gpath.write_text(yaml.safe_dump(gt, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"video": str(video), "zones": str(zpath), "gt": str(gpath), "frames": n}
