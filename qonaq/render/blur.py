"""Обезличивание: размытие верхней части bbox человека (голова/лицо). SPEC §9 — обязательное свойство.

Используется пикселизация + размытие: устойчиво к «восстановлению» и дёшево на CPU.
Блюрим ВСЕ детекции людей, включая неуверенные (порог blur_conf ниже порога трекинга).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np


def head_region(xyxy: Sequence[float], top_fraction: float = 0.38, expand: float = 0.08) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    # Сидящие/обрезанные bbox почти квадратные: голова занимает большую долю высоты.
    frac = top_fraction if h > 1.6 * w else max(top_fraction, 0.5)
    return (int(x1 - expand * w), int(y1 - expand * h), int(x2 + expand * w), int(y1 + frac * h))


def blur_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    # до ~6 блоков по меньшей стороне — лицо неузнаваемо при любом размере
    k = max(1, min(w, h) // 6)
    small = cv2.resize(roi, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    img[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur_people(img: np.ndarray, boxes: Iterable[Sequence[float]], top_fraction: float = 0.38) -> np.ndarray:
    """Размывает головы на месте (in-place) и возвращает тот же массив."""
    for b in boxes:
        blur_region(img, *head_region(b, top_fraction))
    return img
