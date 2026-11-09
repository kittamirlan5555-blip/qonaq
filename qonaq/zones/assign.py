"""Привязка людей к зонам (SPEC §9a п.2): стратегии якоря bottom_center / center / overlap / hips.

Чистая геометрия без OpenCV-окон: полигоны и bbox — обычные числа, поэтому легко тестируется.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qonaq.zones.models import Zone, ZoneSet

BBox = tuple[float, float, float, float]
_GRID = 7  # сетка 7x7 точек для оценки доли bbox внутри полигона


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Векторный ray casting. points: (N,2), polygon: (M,2) -> bool (N,)."""
    x, y = points[:, 0], points[:, 1]
    px, py = polygon[:, 0], polygon[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        cond = (py[i] > y) != (py[j] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
        inside ^= cond & (x < xint)
        j = i
    return inside


def point_in_polygon(pt: tuple[float, float], polygon: np.ndarray) -> bool:
    return bool(points_in_polygon(np.array([pt], dtype=np.float32), polygon)[0])


def bbox_overlap_fraction(box: BBox, polygon: np.ndarray) -> float:
    """Доля площади bbox, лежащая внутри полигона (оценка по сетке точек)."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    xs = np.linspace(x1, x2, _GRID * 2 + 1)[1::2]
    ys = np.linspace(y1, y2, _GRID * 2 + 1)[1::2]
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return float(points_in_polygon(pts, polygon).mean())


def anchor_point(box: BBox, strategy: str, hips: tuple[float, float] | None = None) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    if strategy == "bottom_center":
        return cx, y2
    if strategy == "center":
        return cx, (y1 + y2) / 2
    if strategy == "hips":
        # Если есть pose-ключевые точки — берём их; иначе приближение: 65% высоты bbox.
        return hips if hips is not None else (cx, y1 + 0.65 * (y2 - y1))
    raise ValueError(f"нет точечного якоря для стратегии {strategy}")


def zone_score(box: BBox, zone: Zone, polygon: np.ndarray, hips: tuple[float, float] | None = None) -> float:
    """Оценка принадлежности человека зоне: 0 — не в зоне, иначе (0..1], больше — «сильнее»."""
    strategy = zone.anchor_strategy
    if strategy == "overlap":
        frac = bbox_overlap_fraction(box, polygon)
        return frac if frac >= zone.overlap_threshold else 0.0
    pt = anchor_point(box, strategy, hips)
    if not point_in_polygon(pt, polygon):
        return 0.0
    # Для точечных якорей оценка — тоже доля перекрытия (для выбора между зонами), но не ниже 0.01.
    return max(bbox_overlap_fraction(box, polygon), 0.01)


class ZoneAssigner:
    """Присваивает каждому человеку максимум одну зону каждого «слоя».

    * зона `ignore` перекрывает остальные: человек, чья нижняя центральная точка в ignore, не считается;
    * среди столов человек отдаётся тому, где перекрытие больше (SPEC §9a п.2);
    * зоны entrance / waiting_area / service_point считаются независимо от столов.
    """

    def __init__(self, zones: ZoneSet):
        self.zones = zones
        self._poly = {z.id: z.np_polygon() for z in zones.zones}
        self._staff_poly = {
            z.id: np.array(z.staff_polygon, dtype=np.float32) for z in zones.zones if z.staff_polygon
        }

    def is_ignored(self, box: BBox) -> bool:
        pt = anchor_point(box, "bottom_center")
        return any(point_in_polygon(pt, self._poly[z.id]) for z in self.zones.zones if z.type == "ignore")

    def assign_one(self, box: BBox, hips: tuple[float, float] | None = None) -> list[str]:
        """Список id зон, в которых находится человек (не более одного стола)."""
        if self.is_ignored(box):
            return []
        out: list[str] = []
        best_table, best_score = None, 0.0
        for z in self.zones.zones:
            if z.type == "ignore":
                continue
            s = zone_score(box, z, self._poly[z.id], hips)
            if s <= 0:
                continue
            if z.type == "table":
                if s > best_score:
                    best_table, best_score = z.id, s
            else:
                out.append(z.id)
        if best_table:
            out.insert(0, best_table)
        return out

    def assign(self, boxes: Sequence[BBox]) -> list[list[str]]:
        return [self.assign_one(b) for b in boxes]

    def in_staff_area(self, box: BBox, zone_id: str) -> bool:
        poly = self._staff_poly.get(zone_id)
        if poly is None:
            return False
        return point_in_polygon(anchor_point(box, "bottom_center"), poly)
