"""Извлечение признаков по индексу датасета с кэшем на диске (data/features/*.npz)."""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from qonaq.datasets.index import load_rows
from qonaq.roles.encoder import OnnxEncoder, crop_box
from qonaq.roles.features import hsv_features

log = logging.getLogger("qonaq.train")

META_KEYS = ("dataset", "venue", "group", "source", "label", "origin", "image")


def _digest(rows: list[dict]) -> str:
    h = hashlib.sha1()
    for r in rows:
        h.update(f"{r['image']}|{r['x1']},{r['y1']},{r['x2']},{r['y2']}|{r['label']}".encode())
    return h.hexdigest()[:12]


def extract(index_csv: Path, encoder: str | None, out_dir: Path, kind: str = "person",
            project_root: Path | None = None) -> dict[str, np.ndarray]:
    """Признаки всех объектов индекса. encoder=None — только HSV (для базы сравнения).
    kind=person — HSV торса/ног; kind=table — HSV всего кропа стола."""
    from qonaq.config import PROJECT_ROOT

    root = project_root or PROJECT_ROOT
    rows = load_rows(index_csv)
    tag = f"{Path(index_csv).stem}_{encoder or 'none'}_{_digest(rows)}"
    cache = out_dir / f"{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    enc = OnnxEncoder(encoder) if encoder else None
    if enc is not None and kind == "table":
        enc.input_hw = (160, 160)
    by_img: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["image"]].append(i)
    emb = np.zeros((len(rows), enc.dim if enc else 0), np.float32)
    hsv = np.zeros((len(rows), 256), np.float32)
    ok = np.zeros(len(rows), bool)
    for n, (img, idxs) in enumerate(by_img.items()):
        frame = cv2.imread(str(root / img))
        if frame is None:
            continue
        crops, keep = [], []
        for i in idxs:
            r = rows[i]
            box = (r["x1"], r["y1"], r["x2"], r["y2"])
            c = crop_box(frame, box)
            if c is None:
                continue
            if kind == "person":
                hsv[i] = hsv_features(frame, box)
            else:
                hsv[i] = table_hsv(c)
            crops.append(c)
            keep.append(i)
            ok[i] = True
        if enc is not None and crops:
            emb[keep] = enc.embed_crops(crops)
        if n % 500 == 0:
            log.info("%s: кадр %d/%d", tag, n, len(by_img))
    out = {"emb": emb, "hsv": hsv, "ok": ok,
           "small": np.array([r["small"] for r in rows], np.int8),
           "w": np.array([r["x2"] - r["x1"] for r in rows], np.int32),
           "h": np.array([r["y2"] - r["y1"] for r in rows], np.int32)}
    for k in META_KEYS:
        out[k] = np.array([r[k] for r in rows])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def table_hsv(crop: np.ndarray) -> np.ndarray:
    """HSV-гистограмма центральной части кропа стола (256 = 16×4×4) + доля «пёстрых» пикселей не нужна —
    простая гистограмма как база сравнения."""
    h, w = crop.shape[:2]
    c = crop[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]
    if c.size == 0:
        c = crop
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    s = float(hist.sum())
    return np.sqrt(hist / s).astype(np.float32) if s > 0 else np.zeros(256, np.float32)
