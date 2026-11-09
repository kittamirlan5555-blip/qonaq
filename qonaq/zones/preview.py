"""Превью: кадр с наложенными зонами (люди на кадре — с блюром)."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from qonaq.detect.base import PERSON, Detection
from qonaq.render.annotate import draw_zones
from qonaq.render.blur import blur_people
from qonaq.zones.models import ZoneSet


def render_preview(frame: np.ndarray, zones: ZoneSet, detections: Sequence[Detection] = (),
                   blur: bool = True) -> np.ndarray:
    img = frame.copy()
    if blur:
        blur_people(img, [d.xyxy for d in detections if d.label == PERSON])
    h, w = img.shape[:2]
    return draw_zones(img, zones.scaled(w, h), font=max(14, w // 80))


def save_preview(path: str | Path, frame: np.ndarray, zones: ZoneSet, detections: Sequence[Detection] = (),
                 blur: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), render_preview(frame, zones, detections, blur))
    return path
