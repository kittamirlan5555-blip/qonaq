"""Калибровка формы персонала: HSV-гистограмма торса + сравнение с профилем (SPEC §4.3)."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

H_BINS, S_BINS, V_BINS = 8, 4, 4
HIST_SIZE = H_BINS * S_BINS * V_BINS


def torso_crop(frame: np.ndarray, xyxy: Sequence[float]) -> np.ndarray | None:
    """Вырез торса: средняя часть bbox (без головы и ног)."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    cy1, cy2 = int(y1 + 0.22 * h), int(y1 + 0.55 * h)
    H, W = frame.shape[:2]
    cx1, cx2 = max(0, cx1), min(W, cx2)
    cy1, cy2 = max(0, cy1), min(H, cy2)
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    return frame[cy1:cy2, cx1:cx2]


def torso_hist(frame: np.ndarray, xyxy: Sequence[float]) -> np.ndarray | None:
    """Нормированная 3D HSV-гистограмма торса (8x4x4 = 128 чисел). V-канал нужен для чёрной/белой формы."""
    crop = torso_crop(frame, xyxy)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else None


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 − расстояние Бхаттачарьи: 1.0 — одинаковые распределения, 0 — непересекающиеся."""
    bc = float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
    return float(max(0.0, 1.0 - np.sqrt(max(0.0, 1.0 - min(bc, 1.0)))))


class UniformProfile:
    def __init__(self, references: Sequence[np.ndarray], threshold: float = 0.55, note: str = ""):
        self.references = [np.asarray(r, dtype=np.float32) for r in references]
        self.threshold = threshold
        self.note = note

    def score(self, hist: np.ndarray | None) -> float:
        """Максимальная похожесть на эталоны формы, 0..1."""
        if hist is None or not self.references:
            return 0.0
        return max(similarity(hist, r) for r in self.references)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "threshold": self.threshold,
            "note": self.note,
            "bins": [H_BINS, S_BINS, V_BINS],
            "references": [[round(float(v), 5) for v in r] for r in self.references],
        }
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=None,
                                       width=200), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> UniformProfile:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        refs = [np.array(r, dtype=np.float32) for r in data.get("references", [])]
        for r in refs:
            if r.size != HIST_SIZE:
                raise ValueError("профиль формы несовместим с текущим размером гистограммы")
        return cls(refs, float(data.get("threshold", 0.55)), data.get("note", ""))
