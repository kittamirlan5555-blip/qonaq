"""NanoDet-Plus через onnxruntime (S2, сравнение детекторов). Apache-2.0, ONNX из релиза github.com/RangiLyu/nanodet.

Только для сравнения в `scripts/bench_detector.py` (EVAL §15): по умолчанию система работает на YOLOX.
Плитки и ROI — те же, что у YOLOX (наследуем `_detect_one`), меняется только один прогон `_infer`.
"""
from __future__ import annotations

import urllib.request

import cv2
import numpy as np

from qonaq.detect.base import COCO_IDS
from qonaq.detect.onnx_yolox import OnnxYoloxDetector, weights_dir

URL = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/{name}.onnx"
MEAN = np.array([103.53, 116.28, 123.675], np.float32)
STD = np.array([57.375, 57.12, 58.395], np.float32)


class OnnxNanoDetDetector(OnnxYoloxDetector):
    def __init__(self, model: str = "nanodet-plus-m-1.5x_416", conf: float = 0.25, iou: float = 0.5,
                 tiles: int = 1, threads: int | None = None, classes: dict[int, str] | None = None, **_):
        import onnxruntime as ort

        path = weights_dir() / f"{model}.onnx"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(name=model), path)  # noqa: S310 — фиксированный https релиза
        self.name, self.conf, self.iou, self.tiles, self.device = model, conf, iou, tiles, "cpu"
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self.size = int(self._sess.get_inputs()[0].shape[2])
        self._classes = dict(classes or COCO_IDS)
        self._cls_idx = np.array(sorted(self._classes), dtype=int)
        pts, strides = [], []
        for s in (8, 16, 32, 64):
            n = int(np.ceil(self.size / s))
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            pts.append(np.stack((xv, yv), 2).reshape(-1, 2) * s)
            strides.append(np.full(n * n, s))
        self._centers = np.concatenate(pts).astype(np.float32)
        self._strides = np.concatenate(strides).astype(np.float32)

    def _infer(self, img: np.ndarray):
        h, w = img.shape[:2]
        x = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        out = self._sess.run(None, {self._input: x})[0][0]
        cls = out[:, :80][:, self._cls_idx]
        best = cls.argmax(1)
        scores = cls[np.arange(len(cls)), best]
        m = scores >= self.conf
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        reg = out[m, 80:].reshape(-1, 4, 8)
        reg = np.exp(reg - reg.max(-1, keepdims=True))
        reg = reg / reg.sum(-1, keepdims=True)
        dist = (reg * np.arange(8, dtype=np.float32)).sum(-1) * self._strides[m, None]
        c = self._centers[m]
        boxes = np.stack([c[:, 0] - dist[:, 0], c[:, 1] - dist[:, 1], c[:, 0] + dist[:, 2], c[:, 1] + dist[:, 3]], 1)
        boxes *= np.array([w / self.size, h / self.size, w / self.size, h / self.size], np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores[m], self._cls_idx[best[m]]
