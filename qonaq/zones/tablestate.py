"""Сигналы состояния стола для слияния в логике (W3): классификатор кропа и посуда детектора.

Третий сигнал — эталон пустого стола — живёт в `qonaq/zones/appearance.py`.
Здесь ничего не решается: только наблюдения True/False/None по каждой зоне-столу.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from qonaq.detect.base import DISH_LABELS, Detection
from qonaq.roles.head import Head
from qonaq.zones.models import ZoneSet

log = logging.getLogger("qonaq.zones")


def table_crop(frame: np.ndarray, bbox: Sequence[float], margin: float = 0.05) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    a, b = int(max(0, x1 - margin * w)), int(max(0, y1 - margin * h))
    c, d = int(min(W, x2 + margin * w)), int(min(H, y2 + margin * h))
    if c - a < 8 or d - b < 8:
        return None
    return frame[b:d, a:c]


class TableStateClassifier:
    """Энкодер + HSV кропа стола + голова clean/dirty/occupied (`models/tables_head.npz`)."""

    def __init__(self, head: Head, encoder=None):
        from qonaq.roles.features import table_hsv

        self.head = head
        self._hsv = table_hsv
        self.use_hsv = bool(head.meta.get("hsv", True))
        enc = head.meta.get("encoder")
        if enc and encoder is None:
            from qonaq.roles.encoder import OnnxEncoder

            encoder = OnnxEncoder(enc, input_hw=(160, 160))
        self.encoder = encoder

    @classmethod
    def load(cls, path: str | Path) -> TableStateClassifier:
        return cls(Head.load(path))

    def proba(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """(N, 3) в порядке head.classes."""
        if not crops:
            return np.zeros((0, len(self.head.classes)), np.float32)
        parts = []
        if self.encoder is not None:
            parts.append(self.encoder.embed_crops(list(crops)))
        if self.use_hsv:
            parts.append(np.stack([self._hsv(c) for c in crops]))
        return self.head.predict_proba(np.concatenate(parts, axis=1))


def _touches(box, people) -> bool:
    x1, y1, x2, y2 = box
    return any(p[0] < x2 and p[2] > x1 and p[1] < y2 and p[3] > y1 for p in people)


class TableSignals:
    """Сигналы `clf` и `dishes` по каждой зоне-столу. Пока люди касаются bbox стола — None (не видно стол)."""

    def __init__(self, zones: ZoneSet, th, classifier: TableStateClassifier | None = None):
        self.th = th
        self.clf = classifier if th.use_table_classifier else None
        self.use_dishes = th.use_dish_detection
        self._zones = {z.id: z for z in zones.by_type("table")}
        self._clf_last: dict[str, tuple[float, bool | None, dict]] = {}
        self.last_proba: dict[str, dict[str, float]] = {}

    def dishes_in(self, zone_id: str, objects: Sequence[Detection]) -> int:
        z = self._zones[zone_id]
        poly = np.array(z.polygon, np.float32)
        n = 0
        for d in objects:
            if d.label not in DISH_LABELS or d.conf < self.th.dish_conf:
                continue
            cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                n += 1
        return n

    def update(self, frame: np.ndarray, ts: float, people: Sequence[Sequence[float]],
               objects: Sequence[Detection]) -> dict[str, list[tuple[str, bool | None]]]:
        out: dict[str, list[tuple[str, bool | None]]] = {}
        due, due_crops = [], []
        for zid, z in self._zones.items():
            box = z.bbox()
            sig: list[tuple[str, bool | None]] = []
            hidden = _touches(box, people)
            if self.use_dishes:
                sig.append(("dishes", None if hidden else self.dishes_in(zid, objects) >= self.th.dish_min_count))
            if self.clf is not None:
                last = self._clf_last.get(zid)
                if hidden:
                    sig.append(("clf", None))
                elif last is not None and ts - last[0] < self.th.table_clf_every_s:
                    sig.append(("clf", last[1]))
                else:
                    c = table_crop(frame, box)
                    if c is not None:
                        due.append(zid)
                        due_crops.append(c)
                    sig.append(("clf", None))  # заполним ниже
            out[zid] = sig
        if due and self.clf is not None:
            probs = self.clf.proba(due_crops)
            cls = self.clf.head.classes
            for zid, p in zip(due, probs, strict=True):
                pd = {c: float(v) for c, v in zip(cls, p, strict=True)}
                self.last_proba[zid] = pd
                val = self._decide(pd)
                self._clf_last[zid] = (ts, val, pd)
                out[zid] = [(n, val if n == "clf" else v) for n, v in out[zid]]
        return out

    def _decide(self, p: dict[str, float]) -> bool | None:
        """Если классификатор считает стол занятым (людей детектор не видит) — не голосуем."""
        dirty, clean, occ = p.get("dirty", 0.0), p.get("clean", 0.0), p.get("occupied", 0.0)
        if occ > max(dirty, clean):
            return None
        r = dirty / max(dirty + clean, 1e-6)
        thr = self.th.table_clf_dirty_p
        if r >= thr:
            return True
        if r <= 1 - thr:
            return False
        return None


def load_table_classifier(path: str | Path) -> TableStateClassifier | None:
    from qonaq.config import resolve_path

    p = resolve_path(str(path))
    if not p.exists():
        log.warning("голова состояния стола не найдена: %s — сигнал clf выключен", p)
        return None
    try:
        return TableStateClassifier.load(p)
    except Exception as e:  # noqa: BLE001
        log.warning("классификатор стола не загрузился (%s) — сигнал clf выключен", e)
        return None
