"""Адаптер Ultralytics YOLO -> интерфейс Detector.

Единственное место в проекте, где импортируется ultralytics (AGPL-3.0, см. DECISIONS.md D-002).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from qonaq.detect.base import COCO_IDS, Detection

log = logging.getLogger("qonaq.detect")


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def weights_dir() -> Path:
    from qonaq.config import PROJECT_ROOT

    d = PROJECT_ROOT / "data" / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


class UltralyticsDetector:
    def __init__(self, model: str = "yolo11n.pt", fallback_models: Sequence[str] = (), conf: float = 0.25,
                 imgsz: int = 960, device: str = "auto"):
        from ultralytics import YOLO

        self.device = pick_device(device)
        self.conf = conf
        self.imgsz = imgsz
        self._model = None
        errors = []
        for name in [model, *fallback_models]:
            path = Path(name)
            if not path.exists() and path.parent == Path("."):
                path = weights_dir() / name  # веса лежат в data/weights (в .gitignore)
            try:
                self._model = YOLO(str(path))
                self.name = Path(name).stem
                break
            except Exception as e:  # noqa: BLE001  — нет сети / нет такой модели
                errors.append(f"{name}: {e}")
                log.warning("модель %s недоступна: %s", name, e)
        if self._model is None:
            raise RuntimeError("не удалось загрузить ни одну модель детекции: " + "; ".join(errors))
        self._classes = sorted(COCO_IDS)

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        results = self._model.predict(list(frames), conf=self.conf, imgsz=self.imgsz, device=self.device,
                                      classes=self._classes, verbose=False)
        out: list[list[Detection]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                out.append([])
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            out.append([
                Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), COCO_IDS[int(k)])
                for b, c, k in zip(xyxy, conf, cls, strict=True) if int(k) in COCO_IDS
            ])
        return out
