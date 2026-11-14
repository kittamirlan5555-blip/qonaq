"""Рендер синтетического ролика по сценарию: фон + фигуры людей из датасетов + «посуда».

    python -m qonaq synth configs/synth/normal_visit.yaml
→ data/videos/synth_<name>.mp4, data/gt/synth_<name>.yaml, configs/zones/synth_<name>.yaml

Фигуры — кропы из data/index/roles.csv (нужны датасеты W1) с мягкой эллиптической маской.
Кадры детерминированы (без шума): одинаковые кадры дают попадания в кэш детекций.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from qonaq.synth.scenario import Scenario, Waypoint, derive_events, dish_intervals

log = logging.getLogger("qonaq.synth")


# --------------------------------------------------------------------------- банк фигур
class SpriteBank:
    def __init__(self, roles_csv: Path, tables_csv: Path | None, root: Path, seed: int = 0):
        from qonaq.datasets.index import load_rows

        self.root = root
        rows = [r for r in load_rows(roles_csv) if not r["small"]]
        rng = np.random.default_rng(seed)
        self._by: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            w, h = r["x2"] - r["x1"], r["y2"] - r["y1"]
            if w < 40 or h < 60:  # крупные кропы: не растягивать мелочь
                continue
            self._by.setdefault((r["label"], r["venue"]), []).append(r)
        for v in self._by.values():
            rng.shuffle(v)
        self._dish_rows: list[dict] = []
        self._cache: dict = {}

    def person(self, kind: str, venues: list[str], idx: int, pose: str) -> tuple[np.ndarray, np.ndarray]:
        """BGR-кроп и альфа-маска. pose=stand предпочитает высокие кропы, sit — приземистые."""
        key = (kind, tuple(venues), idx, pose)
        if key in self._cache:
            return self._cache[key]
        pool = [r for v in venues for r in self._by.get((kind, v), [])]
        if not pool:
            raise RuntimeError(f"нет фигур {kind} из залов {venues}: python -m qonaq data index")
        ar = np.array([(r["y2"] - r["y1"]) / max(1, r["x2"] - r["x1"]) for r in pool])
        cand = [r for r, a in zip(pool, ar, strict=True) if (a >= 1.8 if pose == "stand" else 0.8 <= a <= 1.6)]
        pool = cand or pool
        r = pool[idx % len(pool)]
        img = cv2.imread(str(self.root / r["image"]))
        crop = img[r["y1"]:r["y2"], r["x1"]:r["x2"]].copy()
        alpha = soft_ellipse(crop.shape[1], crop.shape[0])
        self._cache[key] = (crop, alpha)
        return crop, alpha


def soft_ellipse(w: int, h: int, feather: float = 0.18) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return np.clip((1.0 - d) / feather, 0, 1).astype(np.float32)


def paste(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray, cx: float, bottom: float, height: float) -> None:
    """Вставка фигуры: низ фигуры в точке (cx, bottom), высота height px."""
    s = height / src.shape[0]
    w, h = max(2, int(src.shape[1] * s)), max(2, int(height))
    im = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(cx - w / 2), int(bottom - h)
    H, W = dst.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    sub_im = im[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    sub_a = a[ya - y0:yb - y0, xa - x0:xb - x0][..., None]
    roi = dst[ya:yb, xa:xb].astype(np.float32)
    dst[ya:yb, xa:xb] = (sub_a * sub_im + (1 - sub_a) * roi).astype(np.uint8)


def draw_dishes(img: np.ndarray, poly: list[list[float]], seed: int) -> None:
    """«Посуда»: несколько тарелок/чашек с тенью и остатками еды в центре стола (рисованные, без датасета)."""
    rng = np.random.default_rng(seed)
    p = np.array(poly, np.float32)
    cx, cy = p.mean(axis=0)
    w = float(p[:, 0].max() - p[:, 0].min())
    r0 = max(6, int(w * 0.09))
    for k in range(4):
        x = int(cx + rng.uniform(-0.25, 0.25) * w)
        y = int(cy + rng.uniform(-0.15, 0.15) * w)
        r = int(r0 * rng.uniform(0.7, 1.1))
        cv2.ellipse(img, (x + 2, y + 3), (r, int(r * 0.7)), 0, 0, 360, (60, 60, 60), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (r, int(r * 0.7)), 0, 0, 360, (235, 235, 240), -1, cv2.LINE_AA)
        if k % 2 == 0:  # остатки еды
            col = tuple(int(c) for c in rng.integers(30, 150, 3))
            cv2.ellipse(img, (x, y), (r // 2, int(r * 0.35)), 15, 0, 360, col, -1, cv2.LINE_AA)
        else:  # чашка
            cv2.circle(img, (x, y), max(3, r // 2), (90, 110, 160), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- геометрия
def zone_anchor(sc: Scenario, w: Waypoint) -> tuple[float, float]:
    if isinstance(w.at, list):
        return float(w.at[0]), float(w.at[1])
    z = next(z for z in sc.zones if z.id == w.at)
    if z.seats:
        s = z.seats[w.seat % len(z.seats)]
        return float(s[0]), float(s[1])
    p = np.array(z.polygon, np.float32)
    return float(p[:, 0].mean()), float(p[:, 1].max())


def position(sc: Scenario, path: list[Waypoint], t: float) -> tuple[float, float, str, bool] | None:
    """Координаты ног актёра в момент t: линейно между точками (идёт), стоит/сидит на месте."""
    prev = None
    for i, w in enumerate(path):
        if w.t > t:
            if prev is None:
                return None
            nxt = w
            # движение: последние min(4, интервал) секунд перед следующей точкой — переход
            move = min(4.0, nxt.t - prev.t)
            x0, y0 = zone_anchor(sc, prev)
            x1, y1 = zone_anchor(sc, nxt)
            if t >= nxt.t - move and prev.visible and nxt.visible and (x0, y0) != (x1, y1):
                a = (t - (nxt.t - move)) / move
                return x0 + a * (x1 - x0), y0 + a * (y1 - y0), "stand", True
            return x0, y0, prev.pose, prev.visible
        prev = path[i]
    if prev is None:
        return None
    x, y = zone_anchor(sc, prev)
    return x, y, prev.pose, prev.visible


def background(sc: Scenario, root: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(root / sc.background.video))
    if not cap.isOpened():
        raise FileNotFoundError(sc.background.video)
    if sc.background.mode == "frame":
        cap.set(cv2.CAP_PROP_POS_MSEC, sc.background.at_s * 1000)
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("фон не читается")
        return f
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for k in np.linspace(0, max(0, n - 1), 25).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def add_sensor_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Шум матрицы, как у настоящей камеры (σ уровней яркости, независимо по кадрам)."""
    if sigma <= 0:
        return img
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def render(sc: Scenario, out_video: Path, root: Path, bank: SpriteBank) -> int:
    from qonaq.render.sink import VideoSink

    bg = background(sc, root)
    dishes = dish_intervals(sc) if sc.dishes else {}
    n = int(sc.duration_s * sc.fps)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    sink = VideoSink(out_video, sc.fps, crf=18)
    noise_rng = np.random.default_rng(12345)
    try:
        for i in range(n):
            t = i / sc.fps
            img = bg.copy()
            for z in sc.zones:
                if any(a <= t < b for a, b in dishes.get(z.id, [])):
                    draw_dishes(img, z.polygon, seed=hash(z.id) % 1000)
            drawn = []
            for a in sc.actors:
                pos = position(sc, a.path, t)
                if pos is None or not pos[3]:
                    continue
                drawn.append((pos[1], a, pos))
            for _, a, (x, y, pose, _) in sorted(drawn, key=lambda d: d[0]):  # дальние первыми
                venues = [sc.staff_venue] if a.kind == "staff" else sc.guest_venues
                crop, alpha = bank.person(a.kind, venues, a.sprite, pose)
                k = 1.0 + sc.scale_k * (y - sc.scale_ref_y) if sc.scale_ref_y is not None else 1.0
                paste(img, crop, alpha, x, y, (sc.sit_h if pose == "sit" else sc.stand_h) * k)
            sink.write(add_sensor_noise(img, sc.sensor_noise, noise_rng))
    finally:
        sink.close()
    return n


def build(scenario_path: Path, root: Path, out_dir: Path | None = None) -> dict:
    sc = Scenario.load(scenario_path)
    name = f"synth_{sc.name}"
    video = (out_dir or root / "data" / "videos") / f"{name}.mp4"
    bank = SpriteBank(root / "data" / "index" / "roles.csv", None, root)
    frames = render(sc, video, root, bank)
    zones_path = root / "configs" / "zones" / f"{name}.yaml"
    bg = background(sc, root)
    zones_path.write_text(yaml.safe_dump({
        "camera_id": name, "frame_size": [bg.shape[1], bg.shape[0]],
        "zones": [{"id": z.id, "type": "table", "polygon": z.polygon} for z in sc.zones]},
        allow_unicode=True, sort_keys=False), encoding="utf-8")
    gt = {"video": video.name, "zones": str(zones_path.relative_to(root).as_posix()),
          "synthetic": True, "scenario": str(scenario_path.as_posix()),
          "alerts_forbidden": sc.alerts_forbidden, "alerts_expected": sc.alerts_expected,
          "events": derive_events(sc), "notes": sc.notes or sc.description}
    gt_path = root / "data" / "gt" / f"{name}.yaml"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(yaml.safe_dump(gt, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    log.info("синтетика %s: %d кадров -> %s; разметка %s", sc.name, frames, video, gt_path)
    return {"video": str(video), "gt": str(gt_path), "zones": str(zones_path), "frames": frames}
