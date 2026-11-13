"""Замороженные ONNX-энкодеры кропов (W2/W3): эмбеддинг человека или стола для лёгкой головы.

Без torch: только onnxruntime. Веса качает `python scripts/fetch_models.py` в data/weights/encoders/.
Лицензии: оба энкодера — из opencv/opencv_zoo, Apache-2.0 (docs/DATA_SOURCES.md).
"""
from __future__ import annotations

import logging
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("qonaq.encoder")

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    url: str
    file: str                 # имя файла после подготовки
    input_hw: tuple[int, int]  # (H, W) входа для кропа
    cut_output: str | None    # имя тензора, на котором обрезать граф (предпоследний слой)
    dim: int
    license: str


ENCODERS: dict[str, EncoderSpec] = {
    # ImageNet MobileNetV2, обрезан до глобального пулинга (1280 признаков). Вход любого размера (GAP).
    "mnv2": EncoderSpec("mnv2", _ZOO + "image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx",
                        "mnv2_gap.onnx", (160, 96), "472", 1280, "Apache-2.0 (opencv_zoo)"),
    # Youtu ReID baseline (ResNet-50 класс), эмбеддинг 768. Специально обучен различать людей по одежде.
    "reid": EncoderSpec("reid", _ZOO + "person_reid_youtureid/person_reid_youtu_2021nov.onnx",
                        "reid_youtu_dyn.onnx", (128, 64), None, 768, "Apache-2.0 (opencv_zoo)"),
}


def encoders_dir() -> Path:
    from qonaq.config import PROJECT_ROOT

    d = PROJECT_ROOT / "data" / "weights" / "encoders"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_encoder(name: str) -> Path:
    spec = ENCODERS[name]
    dst = encoders_dir() / spec.file
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    raw = encoders_dir() / ("raw_" + Path(spec.url).name)
    if not raw.exists():
        log.info("скачиваю энкодер %s ...", spec.url)
        tmp = raw.with_suffix(".part")
        urllib.request.urlretrieve(spec.url, tmp)  # noqa: S310 — фиксированный адрес opencv_zoo
        tmp.replace(raw)
    import onnx  # нужен только для подготовки (обрезка классификатора)

    m = onnx.load(str(raw))
    if spec.cut_output:
        # оставить только узлы, от которых зависит cut_output (выкинуть классификатор)
        needed: set[str] = {spec.cut_output}
        nodes = []
        for n in reversed(m.graph.node):
            if any(o in needed for o in n.output):
                nodes.append(n)
                needed.update(n.input)
        nodes.reverse()
        del m.graph.node[:]
        m.graph.node.extend(nodes)
        del m.graph.output[:]
        m.graph.output.append(onnx.helper.make_tensor_value_info(spec.cut_output, onnx.TensorProto.FLOAT, None))
        used = {i for n in nodes for i in n.input}
        inits = [t for t in m.graph.initializer if t.name in used]
        del m.graph.initializer[:]
        m.graph.initializer.extend(inits)
    # вход любого размера: после GlobalAveragePool размер не важен
    dims = m.graph.input[0].type.tensor_type.shape.dim
    for i, nm in ((0, "batch"), (2, "height"), (3, "width")):
        dims[i].ClearField("dim_value")
        dims[i].dim_param = nm
    del m.graph.value_info[:]
    onnx.save(m, str(dst))
    raw.unlink()
    return dst


class OnnxEncoder:
    """Эмбеддинги кропов батчами. `embed(frame, boxes)` -> (N, dim), L2-нормированные."""

    def __init__(self, name: str = "mnv2", threads: int | None = None, input_hw: tuple[int, int] | None = None):
        import onnxruntime as ort

        self.spec = ENCODERS[name]
        self.name = name
        self.input_hw = input_hw or self.spec.input_hw
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(ensure_encoder(name)), sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self._inp = self._sess.get_inputs()[0].name
        self.dim = self.spec.dim

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        h, w = self.input_hw
        im = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if crop.shape[0] > h else cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = (im - IMAGENET_MEAN) / IMAGENET_STD
        return im.transpose(2, 0, 1)

    def embed_crops(self, crops: Sequence[np.ndarray], batch: int = 32) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for i in range(0, len(crops), batch):
            x = np.stack([self._prep(c) for c in crops[i:i + batch]]).astype(np.float32)
            y = self._sess.run(None, {self._inp: x})[0].reshape(len(x), -1)
            out.append(y)
        e = np.concatenate(out).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def crop_box(frame: np.ndarray, xyxy: Sequence[float], margin: float = 0.08) -> np.ndarray | None:
    """Кроп с небольшим полем вокруг bbox (W1: «bbox с небольшим полем»)."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    a, b = int(max(0, x1 - margin * w)), int(max(0, y1 - margin * h))
    c, d = int(min(W, x2 + margin * w)), int(min(H, y2 + margin * h))
    if c - a < 4 or d - b < 4:
        return None
    return frame[b:d, a:c]
