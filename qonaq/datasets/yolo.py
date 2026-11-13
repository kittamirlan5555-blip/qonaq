"""Чтение экспорта Roboflow в формате YOLOv8 и группировка дублей по исходнику."""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_RF = re.compile(r"^(?P<src>.+?)_(?:jpg|jpeg|png|bmp)\.rf\.[0-9a-fA-F]+$", re.IGNORECASE)
# Родные split-ы Roboflow. Предпочтение при дедупликации: valid/test обычно без аугментаций.
SPLIT_PREFERENCE = ("valid", "test", "train")


def source_key(filename: str) -> str:
    """`cam1_frame_0001_jpg.rf.<hash>.jpg` -> `cam1_frame_0001`. Аугментации одного кадра дают один ключ."""
    stem = Path(filename).stem
    m = _RF.match(stem)
    return m.group("src") if m else stem


@dataclass
class YoloBox:
    cls_name: str
    x1: float
    y1: float
    x2: float
    y2: float  # нормированные 0..1


@dataclass
class YoloImage:
    path: Path
    split: str
    source: str
    boxes: list[YoloBox]


def class_names(ds_dir: Path) -> list[str]:
    data = yaml.safe_load((ds_dir / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def parse_label_file(path: Path, names: list[str]) -> list[YoloBox]:
    out: list[YoloBox] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:  # полигон: x y x y ...
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if 0 <= cls < len(names):
            out.append(YoloBox(names[cls], max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)))
    return out


def iter_images(ds_dir: Path) -> Iterator[YoloImage]:
    names = class_names(ds_dir)
    for split in ("train", "valid", "test"):
        img_dir = ds_dir / split / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lbl = ds_dir / split / "labels" / (p.stem + ".txt")
            yield YoloImage(p, split, source_key(p.name), parse_label_file(lbl, names))


def dedupe(images: list[YoloImage]) -> list[YoloImage]:
    """Один кадр на исходник. Предпочитаем valid/test (Roboflow аугментирует только train)."""
    best: dict[str, YoloImage] = {}
    rank = {s: i for i, s in enumerate(SPLIT_PREFERENCE)}
    for im in images:
        cur = best.get(im.source)
        if cur is None or rank.get(im.split, 9) < rank.get(cur.split, 9):
            best[im.source] = im
    return [best[k] for k in sorted(best)]
