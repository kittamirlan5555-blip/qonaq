"""Автопредложение зон столов (SPEC §3.1): детекции `dining table` (+`chair`) по N кадрам ->
кластеризация устойчивых -> расширение на стулья -> черновые полигоны T01… слева-направо, сверху-вниз.

Кластеризация и геометрия — чистые функции от списков Detection (тестируются без видео и модели).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qonaq.detect.base import CHAIR, TABLE, Detection, Detector
from qonaq.zones.models import Zone, ZoneSet

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@dataclass
class Cluster:
    boxes: list[Box]
    frames: set[int]
    conf: float = 0.0

    @property
    def box(self) -> Box:
        return tuple(float(v) for v in np.median(np.array(self.boxes), axis=0))


def overlap_min(a: Box, b: Box) -> float:
    """Пересечение, делённое на площадь меньшей рамки (1.0 — одна рамка целиком внутри другой)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(1e-9, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def cluster_boxes(per_frame: Sequence[Sequence[Detection]], label: str, iou_thr: float = 0.3,
                  min_support: float = 0.25, min_conf: float = 0.2) -> list[Cluster]:
    """Жадная кластеризация рамок одного класса по IoU с медианной рамкой кластера."""
    clusters: list[Cluster] = []
    n_frames = len(per_frame)
    for fi, dets in enumerate(per_frame):
        for d in sorted((d for d in dets if d.label == label and d.conf >= min_conf), key=lambda d: -d.conf):
            best, best_iou = None, iou_thr
            for c in clusters:
                v = iou(d.xyxy, c.box)
                if v >= best_iou:
                    best, best_iou = c, v
            if best is None:
                clusters.append(Cluster([d.xyxy], {fi}, d.conf))
            else:
                best.boxes.append(d.xyxy)
                best.frames.add(fi)
                best.conf = max(best.conf, d.conf)
    need = max(2 if n_frames >= 4 else 1, int(np.ceil(min_support * n_frames)))
    stable = [c for c in clusters if len(c.frames) >= need]
    # сливаем дубли (два кластера об одном столе)
    stable.sort(key=lambda c: -len(c.frames))
    merged: list[Cluster] = []
    for c in stable:
        if any(overlap_min(c.box, m.box) > 0.45 for m in merged):
            continue
        merged.append(c)
    return merged


def scale_box(b: Box, k: float) -> Box:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w, h = (b[2] - b[0]) * k, (b[3] - b[1]) * k
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def expand_table(table: Box, chairs: Sequence[Box], expand: float = 1.4, chair_reach: float = 1.6,
                 max_expand: float = 1.9) -> Box:
    """Стол x1.4 (захват стульев), плюс объединение со стульями рядом, но не больше max_expand."""
    box = scale_box(table, expand)
    reach = scale_box(table, chair_reach)
    limit = scale_box(table, max_expand)
    x1, y1, x2, y2 = box
    for c in chairs:
        cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
        if reach[0] <= cx <= reach[2] and reach[1] <= cy <= reach[3]:
            x1, y1, x2, y2 = min(x1, c[0]), min(y1, c[1]), max(x2, c[2]), max(y2, c[3])
    return (max(x1, limit[0]), max(y1, limit[1]), min(x2, limit[2]), min(y2, limit[3]))


def reading_order(boxes: Sequence[Box]) -> list[int]:
    """Порядок «сверху вниз, слева направо»: строки группируются по близости центров по Y."""
    if not boxes:
        return []
    cy = [(b[1] + b[3]) / 2 for b in boxes]
    cx = [(b[0] + b[2]) / 2 for b in boxes]
    row_h = float(np.median([b[3] - b[1] for b in boxes])) * 0.5
    order = sorted(range(len(boxes)), key=lambda i: cy[i])
    rows: list[list[int]] = [[order[0]]]
    for i in order[1:]:
        ref = float(np.mean([cy[j] for j in rows[-1]]))
        if abs(cy[i] - ref) <= row_h:
            rows[-1].append(i)
        else:
            rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda j: cx[j])]


def suggest_from_detections(per_frame: Sequence[Sequence[Detection]], frame_size: tuple[int, int],
                            camera_id: str = "", expand: float = 1.4, min_support: float = 0.25) -> ZoneSet:
    w, h = frame_size
    tables = cluster_boxes(per_frame, TABLE, min_support=min_support)
    chairs = [c.box for c in cluster_boxes(per_frame, CHAIR, min_support=min_support)]
    # слишком большие «столы» (барная стойка во весь кадр) — отбрасываем
    tables = [t for t in tables if (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]) < 0.2 * w * h]
    expanded: list[Box] = []
    for t in sorted(tables, key=lambda c: -len(c.frames)):  # устойчивые столы — первыми
        x1, y1, x2, y2 = expand_table(t.box, chairs, expand)
        box = (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))
        # после расширения два кластера об одном столе почти совпадают — оставляем более устойчивый
        if any(overlap_min(box, e) > 0.6 for e in expanded):
            continue
        expanded.append(box)
    zones = []
    for n, i in enumerate(reading_order(expanded), start=1):
        x1, y1, x2, y2 = expanded[i]
        zones.append(Zone(id=f"T{n:02d}", type="table",
                          polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
    return ZoneSet(camera_id=camera_id, frame_size=(w, h), zones=zones)


def sample_frames(video: str | Path, n: int = 30) -> list[np.ndarray]:
    """N кадров, равномерно по длине видео (для картинки — один кадр)."""
    path = str(video)
    if Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(path)
        if img is None:
            raise OSError(f"не удалось прочитать изображение: {path}")
        return [img]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"не удалось открыть видео: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    if total > 0:
        step = max(1, total // n)
        idx = 0
        want = set(range(0, total, step))
        while True:
            if not cap.grab():
                break
            if idx in want:
                ok, f = cap.retrieve()
                if ok:
                    frames.append(f)
                if len(frames) >= n:
                    break
            idx += 1
    else:
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    cap.release()
    if not frames:
        raise OSError(f"в видео нет кадров: {path}")
    return frames


def suggest_zones(video: str | Path, detector: Detector, n_frames: int = 30, camera_id: str | None = None,
                  expand: float = 1.4) -> tuple[ZoneSet, np.ndarray, list[Detection]]:
    """Возвращает (зоны, опорный кадр, детекции опорного кадра — для блюра превью)."""
    frames = sample_frames(video, n_frames)
    per_frame = [detector.detect([f])[0] for f in frames]
    h, w = frames[0].shape[:2]
    zs = suggest_from_detections(per_frame, (w, h), camera_id or Path(video).stem, expand)
    mid = len(frames) // 2
    return zs, frames[mid], list(per_frame[mid])
