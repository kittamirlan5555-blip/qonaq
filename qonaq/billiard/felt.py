"""Видео-слой бильярдного стола (B2): что происходит на сукне — по разности кадров внутри полигона сукна.

Полигон из 4 точек (углы игровой поверхности) выпрямляется перспективой в прямоугольник шириной felt_width
(шар — ~ball_diameter_frac ширины), иначе — ограничивающий прямоугольник + маска. Дальше на каждом кадре:

1. Свет: медиана яркости сукна. Скачок > global_shift относительно прошлого кадра — смена света (лампу
   включили/выключили, мигнул общий свет) → кадр не годится (motion=None, global_change). Медленное мерцание
   ламп — выравнивание яркости (кадр умножается на отношение медиан) перед разностью.
2. Разность с прошлым кадром > diff_px → морфология → связные пятна. Пятно касается края сукна и крупнее
   трёх шаров, или крупнее ball_area_max шаров, или вытянутое (кий) — «крупный объект через край»: человек
   наклонился, рука, кий, тень, проходящий мимо (intrusion). Мелкие пятна рядом с крупным — его же обрывки.
3. Оставшиеся пятна размером ball_area_min…ball_area_max шаров — катящиеся шары → motion.
4. Лампа над столом: яркость сукна ≥ lamp_on_luma и ≥ lamp_ratio от максимума, виденного раньше.
5. Стол накрыт: доля пикселей цвета сукна (тон, запомненный при горящей лампе) < covered_felt_frac.
6. Шары на сукне (вспомогательно): мелкие пятна, далёкие по цвету от сукна, не у края.

Ничего про людей и роли: решение «стол в игре» принимает qonaq/logic/billiard.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from qonaq.config import BilliardThresholds
from qonaq.logic.base import FeltSignal


class FeltAnalyzer:
    def __init__(self, polygon: list[tuple[float, float]], frame_size: tuple[int, int],
                 th: BilliardThresholds | None = None):
        self.th = th or BilliardThresholds()
        pts = np.float32(polygon)
        fw = self.th.felt_width
        mg = max(4, int(round(self.th.margin_frac * fw)))  # полоса вокруг сукна (борта): через неё входят люди
        self.mg = mg
        if len(pts) == 4:
            top = np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])
            side = np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])
            fh = max(16, int(round(fw * min(top, side) / max(top, side))))
            w, h = (fw, fh) if top >= side else (fh, fw)
            dst = np.float32([[mg, mg], [mg + w, mg], [mg + w, mg + h], [mg, mg + h]])
            self.M = cv2.getPerspectiveTransform(pts, dst)
            self.size = (w + 2 * mg, h + 2 * mg)
            self.mask = np.zeros((self.size[1], self.size[0]), np.uint8)
            self.mask[mg:mg + h, mg:mg + w] = 255
            self.felt_long = float(max(w, h))
        else:  # произвольный многоугольник: масштабируем ограничивающий прямоугольник с полосой
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            k = fw / max(1.0, float(max(x1 - x0, y1 - y0)))
            S = np.float32([[k, 0, mg - k * x0], [0, k, mg - k * y0]])
            self.M = np.vstack([S, [0, 0, 1]]).astype(np.float64)
            self.size = (int((x1 - x0) * k) + 2 * mg, int((y1 - y0) * k) + 2 * mg)
            self.mask = np.zeros((self.size[1], self.size[0]), np.uint8)
            cv2.fillPoly(self.mask, [cv2.transform(pts.reshape(-1, 1, 2), S).reshape(-1, 2).astype(np.int32)], 255)
            self.felt_long = float(fw)
        self.band = cv2.bitwise_not(self.mask)  # всё вне сукна в кропе — борта и пол рядом
        self.inner = cv2.erode(self.mask, np.ones((5, 5), np.uint8))
        self.ball_d = max(2.0, self.th.ball_diameter_frac * self.felt_long)
        self.ball_area = np.pi * (self.ball_d / 2) ** 2
        self._prev: np.ndarray | None = None
        self._prev_med: float | None = None
        self._bg: np.ndarray | None = None
        self._rest: np.ndarray | None = None  # сукно в покое (для проверки «шары легли по-другому»)
        self._rest_busy = np.zeros(self.mask.shape, bool)
        self._quiet = 0
        self._active = False
        self.last_layout_areas: list[float] = []  # фон пустого стола (медленное среднее) — чтобы видеть руку целиком
        self._lamp_ref = 0.0
        self._band_ref = 0.0
        self._hue_ref: float | None = None
        self._hue_obs: list[float] = []
        self.last_debug: dict = {}

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M, self.size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    def update(self, frame: np.ndarray) -> FeltSignal:
        th = self.th
        bgr = self._crop(frame)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        col = cv2.GaussianBlur(bgr, (3, 3), 0).astype(np.float32)  # разность — по цвету: тёмный рукав на
        inside = self.mask > 0  # синем сукне в оттенках серого почти не виден
        med = float(np.median(gray[inside]))
        band_med = float(np.median(gray[self.band > 0])) if self.band.any() else med
        self._lamp_ref = max(self._lamp_ref * 0.9999, med)
        self._band_ref = max(self._band_ref * 0.9999, band_med)
        lamp_on = med >= th.lamp_on_luma and med >= th.lamp_ratio * self._lamp_ref

        # цвет сукна: тон запоминаем при горящей лампе (медиана первых кадров); «накрыт» — сукна не видно
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        covered = None
        sat = hsv[..., 1][inside]
        if lamp_on and self._hue_ref is None:
            sel = sat > 60
            if sel.mean() > 0.5:
                self._hue_obs.append(float(np.median(hsv[..., 0][inside][sel])))
            if len(self._hue_obs) >= 5:
                self._hue_ref = float(np.median(self._hue_obs))
        bright = band_med >= th.lamp_ratio * self._band_ref  # вокруг стола светло — можно судить о цвете
        if self._hue_ref is not None and (lamp_on or bright):
            dh = np.abs(hsv[..., 0][inside].astype(np.float32) - self._hue_ref)
            dh = np.minimum(dh, 180 - dh)
            felt = (dh <= 12) & (sat > 45)
            covered = bool(felt.mean() < th.covered_felt_frac)
            if covered:  # сукно закрыто — о лампе судим по бортам вокруг
                lamp_on = bright

        prev, prev_med = self._prev, self._prev_med
        self._prev, self._prev_med = col, med
        if prev is None or prev_med is None:
            return FeltSignal(motion=None, lamp_on=lamp_on, covered=covered)
        if abs(med - prev_med) > th.global_shift * max(prev_med, 10.0):
            self._bg = None  # свет сменился — фон и снимок покоя заново
            self._rest = None
            return FeltSignal(motion=None, global_change=True, lamp_on=lamp_on, covered=covered)
        if covered:
            return FeltSignal(motion=None, lamp_on=lamp_on, covered=True)
        cur = col * (prev_med / med) if med > 1 else col  # мерцание: выравниваем яркость
        # Крупные объекты через край сукна — по отличию от фона: рука, замершая над столом, видна целиком,
        # а в разности соседних кадров от неё остаются мелкие обрывки по краям, похожие на шары.
        if self._bg is None:
            self._bg = cur.copy()
        fg = (np.abs(cur - self._bg).max(axis=2) > th.fg_px).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        nf, flab, _, _ = cv2.connectedComponentsWithStats(fg, 8)
        cross = np.unique(flab[(self.band > 0) & (fg > 0)])
        intruder = np.isin(flab, cross[cross > 0]) if nf > 1 and cross.size else np.zeros(fg.shape, bool)
        a_bg = np.where(fg > 0, th.bg_rate_fg, th.bg_rate)[..., None]
        self._bg += a_bg * (cur - self._bg)

        d = np.abs(cur - prev).max(axis=2)
        # порог по шуму: сильный шум матрицы/сжатия (тёмный зал, плохой субпоток) даёт пятна размером с шар
        noise = 1.4826 * float(np.median(d[inside]))
        thr = max(th.diff_px, th.noise_k * noise)
        diff = (d > thr).astype(np.uint8) * 255
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(diff, 8)
        crossing = set(np.unique(lab[(self.band > 0) & (diff > 0)]).tolist()) - {0}
        large = (intruder.astype(np.uint8) * 255)
        small: list[int] = []
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            if i in crossing:  # объект пересекает край сукна: человек, рука, кий, тень — не шар
                large[lab == i] = 255
                continue
            a = int(((lab[y:y + h, x:x + w] == i) & inside[y:y + h, x:x + w]).sum()) / self.ball_area
            if a < th.ball_area_min:
                continue
            elong = max(w, h) / max(1, min(w, h))
            if a > th.ball_area_max or (elong > 5.0 and max(w, h) > 4 * self.ball_d):
                large[lab == i] = 255
            else:
                small.append(i)
        intrusion = bool((large & self.mask).any())
        if large.any() and small:  # обрывки крупного объекта рядом с ним — не шары
            k = int(2 * self.ball_d) * 2 + 1
            near = cv2.dilate(large, np.ones((k, k), np.uint8)) > 0
            small = [i for i in small if not near[lab == i].any()]
        score = float(sum(stats[i][4] for i in small) / self.ball_area)
        balls = self._count_balls(bgr) if lamp_on else None
        layout = self._layout(cur, bool(small), large, thr)
        self.last_debug = {"thr": round(thr, 1), "med": med, "small": len(small), "intrusion": intrusion,
                           "score": score, "layout": layout}
        return FeltSignal(motion=bool(small), intrusion=intrusion, lamp_on=lamp_on, covered=covered,
                          balls=balls, score=round(score, 2), layout_changed=layout)

    def _layout(self, cur: np.ndarray, moving: bool, large: np.ndarray, thr: float) -> bool | None:
        """Проверка эпизода: после движения шары легли по-другому? Снимок «стол в покое» до и после.

        Удар всегда переставляет шары (хотя бы биток). Артефакты сжатия, мерцание, рука над столом — нет:
        после них сукно снова как было. Возвращает True/False один раз — на первом спокойном кадре после
        движения (rest_frames кадров без движения); иначе None.
        """
        th = self.th
        self._quiet = 0 if moving else self._quiet + 1
        if moving:
            self._active = True
        if self._quiet < th.rest_frames:
            return None
        k = int(2 * self.ball_d) * 2 + 1
        busy = cv2.dilate(large, np.ones((k, k), np.uint8)) > 0  # под рукой/человеком не сравниваем
        verdict = None
        if self._active and self._rest is not None:
            d = np.abs(cur - self._rest).max(axis=2)
            ch = ((d > max(th.layout_px, thr)) & (self.inner > 0) & ~busy & ~self._rest_busy).astype(np.uint8)
            ch = cv2.morphologyEx(ch, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            n, _, st, _ = cv2.connectedComponentsWithStats(ch, 8)
            areas = st[1:, 4] / self.ball_area if n > 1 else np.array([])
            verdict = bool((areas >= th.layout_min_area).any())
            self.last_layout_areas = sorted(areas.round(2).tolist(), reverse=True)[:5]
        if self._active and self._rest is None:
            verdict = None
        if not busy.any() or self._rest is None or self._active:
            self._rest, self._rest_busy = cur.copy(), busy
        self._active = False
        return verdict

    def _count_balls(self, bgr: np.ndarray) -> int:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref = np.median(lab[self.inner > 0], axis=0)
        dist = np.linalg.norm(lab - ref, axis=2)
        m = ((dist > 40) & (self.inner > 0)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        a = stats[1:, 4] / self.ball_area if n > 1 else np.array([])
        return int(((a >= 0.3) & (a <= 4.0)).sum())


class FeltSet:
    """Анализаторы всех зон billiard_table одной камеры."""

    def __init__(self, zones, frame_size: tuple[int, int], th: BilliardThresholds | None = None):
        self.analyzers = {z.id: FeltAnalyzer(z.polygon, frame_size, th) for z in zones.by_type("billiard_table")}

    def update(self, frame: np.ndarray) -> dict[str, FeltSignal]:
        return {zid: a.update(frame) for zid, a in self.analyzers.items()}
