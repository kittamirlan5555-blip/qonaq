"""Признаки кропа человека: эмбеддинг энкодера + HSV-гистограммы торса и ног (W2)."""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from qonaq.roles.uniform import H_BINS, HIST_SIZE, S_BINS, V_BINS, torso_hist

HSV_DIM = 2 * HIST_SIZE


def legs_hist(frame: np.ndarray, xyxy: Sequence[float]) -> np.ndarray | None:
    """Гистограмма нижней части (брюки/фартук): 55–90% высоты bbox, центральные 50% ширины."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    a, c = max(0, int(x1 + 0.25 * w)), min(W, int(x2 - 0.25 * w))
    b, d = max(0, int(y1 + 0.55 * h)), min(H, int(y1 + 0.90 * h))
    if c - a < 4 or d - b < 4:
        return None
    hsv = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256]).ravel()
    s = float(hist.sum())
    return (hist / s).astype(np.float32) if s > 0 else None


def hsv_features(frame: np.ndarray, xyxy: Sequence[float]) -> np.ndarray:
    """256 чисел: торс + ноги; отсутствующая часть — нули. Корень (Хеллингер) — чтобы линейная голова
    работала с гистограммами как с расстоянием Бхаттачарьи."""
    out = np.zeros(HSV_DIM, np.float32)
    t = torso_hist(frame, xyxy)
    if t is not None:
        out[:HIST_SIZE] = np.sqrt(t)
    lg = legs_hist(frame, xyxy)
    if lg is not None:
        out[HIST_SIZE:] = np.sqrt(lg)
    return out
