"""Маска неподвижных ложных «людей» (D-044, EVAL §26): детекция на одном месте без движения десятки минут.

Детектор видит «человека» в наклейках на стене, в постере, в манекене — рамка стоит пиксель в пиксель весь час, и
для логики это «сидящий гость», который никогда не уходит. Живой человек, даже сидящий час за столом, двигается:
поднимает стакан, поворачивается — рамка «гуляет» на проценты своего размера за минуты.

Правило: якорь — рамка детекции. Детекции, совпадающие с якорем (IoU ≥ `iou`), продлевают его, пока ни одна
координата не ушла от исходной больше чем на `drift` диагонали рамки; ушла — якорь начинается заново с текущей рамки.
Якорь, простоявший `static_s` без движения (с пропусками не дольше `gap_s`), — маска: совпадающие с ним детекции
выбрасываются до логики. Шевельнулось — маска снимается сразу.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from qonaq.detect.base import Detection

log = logging.getLogger("qonaq.static")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class _Anchor:
    ref: tuple[float, float, float, float]
    since: float
    last: float
    masked: bool = False


class StaticMask:
    def __init__(self, static_s: float = 600.0, drift: float = 0.05, iou: float = 0.8, gap_s: float = 30.0):
        self.static_s, self.drift, self.iou, self.gap_s = static_s, drift, iou, gap_s
        self.anchors: list[_Anchor] = []

    def _moved(self, a: _Anchor, box) -> bool:
        x1, y1, x2, y2 = a.ref
        diag = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
        return max(abs(u - v) for u, v in zip(a.ref, box, strict=True)) > self.drift * diag

    def filter(self, ts: float, persons: list[Detection]) -> list[Detection]:
        if self.static_s <= 0:
            return persons
        self.anchors = [a for a in self.anchors if ts - a.last <= self.gap_s]
        keep: list[Detection] = []
        used: set[int] = set()
        for d in persons:
            box = d.xyxy
            # Один якорь на место: рамка, задевающая якорь (IoU ≥ 0,3), — тот же предмет/человек. Совпала плотно —
            # продлевает; сдвинулась — сбрасывает его (иначе дрожащий человек «расползается» по нескольким якорям,
            # и один из них случайно простоит static_s).
            best, best_iou = None, 0.3
            for i, a in enumerate(self.anchors):
                if i in used:
                    continue
                v = _iou(a.ref, box)
                if v >= best_iou:
                    best, best_iou = i, v
            if best is None:
                self.anchors.append(_Anchor(tuple(box), ts, ts))
                used.add(len(self.anchors) - 1)
                keep.append(d)
                continue
            used.add(best)
            a = self.anchors[best]
            if best_iou < self.iou or self._moved(a, box):
                if a.masked:
                    log.info("маска снята: %s", [round(v) for v in a.ref])
                a.ref, a.since, a.masked = tuple(box), ts, False
            a.last = ts
            if not a.masked and ts - a.since >= self.static_s:
                a.masked = True
                log.info("неподвижная детекция %.0f с — маска: %s", ts - a.since, [round(v) for v in a.ref])
            if not a.masked:
                keep.append(d)
        return keep

    @property
    def masked_boxes(self) -> list[tuple[float, float, float, float]]:
        return [a.ref for a in self.anchors if a.masked]
