"""Интерфейс детектора. Весь остальной код знает только про Detection и Detector.

ВАЖНО (DECISIONS.md, D-002): Ultralytics под AGPL-3.0. Никакие объекты ultralytics не должны
выходить за пределы адаптера `qonaq.detect.ultra`, чтобы детектор можно было заменить.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

PERSON, CHAIR, TABLE = "person", "chair", "dining table"
COCO_IDS = {0: PERSON, 56: CHAIR, 60: TABLE}


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    label: str  # person | chair | dining table

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@runtime_checkable
class Detector(Protocol):
    name: str
    device: str

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Батч кадров BGR -> по списку детекций на кадр."""
        ...


class FakeDetector:
    """Детектор для тестов: отдаёт заранее заданные детекции по номеру вызова."""

    name = "fake"
    device = "cpu"

    def __init__(self, script: Sequence[Sequence[Detection]] | None = None):
        self.script = list(script or [])
        self.calls = 0

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        out = []
        for _ in frames:
            dets = list(self.script[self.calls % len(self.script)]) if self.script else []
            out.append(dets)
            self.calls += 1
        return out


def ultralytics_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("ultralytics") is not None and importlib.util.find_spec("torch") is not None


def create_detector(cfg) -> Detector:
    """Фабрика по конфигу (qonaq.config.DetectorConfig).

    backend=auto: ultralytics, если установлен (стек по умолчанию из CLAUDE.md), иначе YOLOX через
    onnxruntime (Apache-2.0, без PyTorch). Порог детектора — минимальный из conf и blur_conf: неуверенные
    детекции людей нужны для блюра, в трекинг идут только >= conf.
    """
    conf = min(cfg.conf, cfg.blur_conf)
    backend = cfg.backend
    if backend == "fake":
        return FakeDetector()
    if backend == "auto":
        backend = "ultralytics" if ultralytics_available() else "onnx"
    if backend == "ultralytics":
        from qonaq.detect.ultra import UltralyticsDetector

        return UltralyticsDetector(model=cfg.model, fallback_models=cfg.fallback_models, conf=conf,
                                   imgsz=cfg.imgsz, device=cfg.device)
    from qonaq.detect.onnx_yolox import OnnxYoloxDetector

    return OnnxYoloxDetector(model=cfg.onnx_model, conf=conf, device=cfg.device, tiles=cfg.onnx_tiles)
