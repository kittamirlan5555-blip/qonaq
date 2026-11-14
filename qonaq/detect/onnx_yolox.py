"""Детектор YOLOX через onnxruntime — без PyTorch и без AGPL.

YOLOX (Megvii) распространяется под Apache-2.0, готовые ONNX-веса лежат в релизах GitHub.
Это вариант по умолчанию там, где ultralytics не установлен, и кандидат для коммерческой версии
(см. DECISIONS.md D-002/D-003).

Режим `tiles=2`: кроме полного кадра, прогоняются две перекрывающиеся половины — мелкие люди на
дальних столах 1080p-камеры находятся заметно лучше ценой x3 по времени. `tiles=N≥3` — сетка плиток;
`roi` — плитки только в области зон столов (S2, EVAL §15).
"""
from __future__ import annotations

import logging
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from qonaq.detect.base import COCO_IDS, Detection

log = logging.getLogger("qonaq.detect")

RELEASE_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{name}.onnx"
KNOWN = ("yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_x")


def weights_dir() -> Path:
    from qonaq.config import PROJECT_ROOT

    d = PROJECT_ROOT / "data" / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_weights(name: str) -> Path:
    """Путь к .onnx; при отсутствии — скачивает из релиза YOLOX на GitHub."""
    p = Path(name)
    if p.exists():
        return p
    stem = p.stem
    dst = weights_dir() / f"{stem}.onnx"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    if stem not in KNOWN:
        raise FileNotFoundError(f"веса не найдены: {name}")
    url = RELEASE_URL.format(name=stem)
    log.info("скачиваю веса %s ...", url)
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — фиксированный https-адрес релиза
    tmp.replace(dst)
    return dst


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Классический NMS. boxes: (N,4) xyxy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


class OnnxYoloxDetector:
    def __init__(self, model: str = "yolox_s", conf: float = 0.25, iou: float = 0.5, device: str = "auto",
                 tiles: int = 1, threads: int | None = None, classes: dict[int, str] | None = None):
        import onnxruntime as ort

        path = ensure_weights(model)
        self.name = path.stem
        self.conf = conf
        self.iou = iou
        self.tiles = tiles
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        self.device = "cpu"
        if device in ("auto", "cuda", "cuda:0") and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
            self.device = "cuda:0"
        elif device == "auto" and "DmlExecutionProvider" in available:  # Windows, любой GPU
            providers.insert(0, "DmlExecutionProvider")
            self.device = "directml"
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input = inp.name
        self.size = int(inp.shape[2])  # квадратный вход 416 или 640
        self._grid, self._stride = self._make_grids(self.size)
        self._classes = dict(classes or COCO_IDS)
        self._cls_idx = np.array(sorted(self._classes), dtype=int)

    @staticmethod
    def _make_grids(size: int) -> tuple[np.ndarray, np.ndarray]:
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def _infer(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Один прогон: возвращает boxes(xyxy, координаты img), scores, class_ids (COCO)."""
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        out = self._sess.run(None, {self._input: blob})[0][0]
        obj = out[:, 4]
        cls = out[:, 5:][:, self._cls_idx] * obj[:, None]
        best = cls.argmax(1)
        scores = cls[np.arange(len(cls)), best]
        m = scores >= self.conf
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        o, scores, best = out[m], scores[m], best[m]
        xy = (o[:, :2] + self._grid[m]) * self._stride[m]
        wh = np.exp(o[:, 2:4]) * self._stride[m]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores, self._cls_idx[best]

    def _detect_one(self, frame: np.ndarray, roi: tuple[float, float, float, float] | None = None
                    ) -> list[Detection]:
        h, w = frame.shape[:2]
        parts = [self._infer(frame)]
        for x0, y0, x1, y1 in tile_boxes(w, h, self.tiles, roi):
            b, s, c = self._infer(frame[y0:y1, x0:x1])
            if len(b):
                # рамки, упирающиеся во внутренний край плитки, обрезаны — их увидит соседняя плитка/полный кадр
                cut = np.zeros(len(b), bool)
                if x0 > 0:
                    cut |= b[:, 0] <= 2
                if x1 < w:
                    cut |= b[:, 2] >= (x1 - x0) - 2
                if y0 > 0:
                    cut |= b[:, 1] <= 2
                if y1 < h:
                    cut |= b[:, 3] >= (y1 - y0) - 2
                b, s, c = b[~cut], s[~cut], c[~cut]
                b = b + np.array([x0, y0, x0, y0], dtype=np.float32)
            parts.append((b, s, c))
        boxes = np.concatenate([p[0] for p in parts])
        scores = np.concatenate([p[1] for p in parts])
        classes = np.concatenate([p[2] for p in parts])
        dets: list[Detection] = []
        for k in np.unique(classes):
            idx = np.where(classes == k)[0]
            for i in nms(boxes[idx], scores[idx], self.iou):
                b = boxes[idx][i]
                dets.append(Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                      float(scores[idx][i]), self._classes[int(k)]))
        return dets

    def detect(self, frames: Sequence[np.ndarray], rois: Sequence | None = None) -> list[list[Detection]]:
        """rois — по прямоугольнику на кадр (x1, y1, x2, y2 в пикселях) или None: плитки только внутри него (S2)."""
        rois = list(rois) if rois is not None else [None] * len(frames)
        return [self._detect_one(f, r) for f, r in zip(frames, rois, strict=True)]


def tile_boxes(w: int, h: int, tiles: int, roi: tuple[float, float, float, float] | None = None,
               overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Плитки (x0, y0, x1, y1) поверх полного кадра. tiles=1 — нет плиток; 2 — две перекрывающиеся половины
    (как раньше); N ≥ 3 — сетка N столбцов × строк по пропорции области. roi — только эта область (зоны столов)."""
    if tiles <= 1:
        return []
    rx0, ry0, rx1, ry1 = (0.0, 0.0, float(w), float(h)) if roi is None else roi
    rx0, ry0 = max(0, int(rx0)), max(0, int(ry0))
    rx1, ry1 = min(w, int(rx1)), min(h, int(ry1))
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw < 32 or rh < 32:
        return []
    if tiles == 2 and roi is None:
        if w <= h * 1.2:
            return []
        tw = min(w, max(h, int(w * 0.6)))
        return [(0, 0, tw, h), (w - tw, 0, w, h)]
    cols = tiles
    rows = max(1, int(round(tiles * rh / max(rw, 1))))
    tw = int(rw / (cols - (cols - 1) * overlap)) if cols > 1 else rw
    th = int(rh / (rows - (rows - 1) * overlap)) if rows > 1 else rh
    out = []
    for r in range(rows):
        y0 = ry0 + (int(r * th * (1 - overlap)) if rows > 1 else 0)
        y0 = min(y0, ry1 - th)
        for c in range(cols):
            x0 = rx0 + (int(c * tw * (1 - overlap)) if cols > 1 else 0)
            x0 = min(x0, rx1 - tw)
            out.append((x0, y0, x0 + tw, y0 + th))
    return out
