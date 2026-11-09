"""Сигнал «стол не убран» по эталону пустого стола (SPEC §9a п.4). Включается флагом
`thresholds.use_table_appearance`.

Эталон — уменьшенный серый снимок зоны стола, когда стол долго FREE и рядом нет людей; обновляется
плавно (свет в зале меняется). Сравнение — средняя абсолютная разница после вычитания средней яркости
(устойчиво к общему изменению освещения). Заметная разница при отсутствии людей = на столе посуда.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from qonaq.zones.models import ZoneSet

SIZE = 48


class TableAppearance:
    def __init__(self, zones: ZoneSet, frame_size: tuple[int, int], diff_threshold: float = 9.0,
                 ref_after_free_s: float = 20.0, margin: float = 0.05):
        self.diff_threshold = diff_threshold
        self.ref_after_free_s = ref_after_free_s
        w, h = frame_size
        self._zones: dict[str, tuple[tuple[int, int, int, int], np.ndarray]] = {}
        for z in zones.by_type("table"):
            x1, y1, x2, y2 = z.bbox()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
            pts = (np.array(z.polygon, dtype=np.float32) - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self._zones[z.id] = ((x1, y1, x2, y2), cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0)
        self._ref: dict[str, np.ndarray] = {}
        self.last_diff: dict[str, float] = {}
        self._margin = margin

    def _patch(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)

    @staticmethod
    def _touches(box: tuple[int, int, int, int], people: Sequence[Sequence[float]]) -> bool:
        x1, y1, x2, y2 = box
        return any(p[0] < x2 and p[2] > x1 and p[1] < y2 and p[3] > y1 for p in people)

    def has_reference(self, zone_id: str) -> bool:
        return zone_id in self._ref

    def update(self, frame: np.ndarray, ts: float, people: Sequence[Sequence[float]],
               states: dict[str, dict[str, Any]]) -> dict[str, bool | None]:
        """zone_id -> True (на столе что-то есть) / False (как эталон) / None (нет данных)."""
        out: dict[str, bool | None] = {}
        for zid, (box, mask) in self._zones.items():
            if self._touches(box, people):
                out[zid] = None  # люди закрывают стол — сравнивать нельзя
                continue
            patch = self._patch(frame, box)
            snap = states.get(zid, {})
            if snap.get("state") == "FREE" and snap.get("state_s", 0.0) >= self.ref_after_free_s:
                ref = self._ref.get(zid)
                self._ref[zid] = patch if ref is None else 0.95 * ref + 0.05 * patch
                out[zid] = False
                continue
            ref = self._ref.get(zid)
            if ref is None:
                out[zid] = None
                continue
            a, b = patch[mask], ref[mask]
            diff = float(np.mean(np.abs((a - a.mean()) - (b - b.mean()))))
            self.last_diff[zid] = diff
            out[zid] = diff > self.diff_threshold
        return out
