"""Индекс объектов датасетов в единой схеме + честные группы для разбиений (W1).

Результат — CSV без изображений: путь к кадру, bbox в пикселях, единая метка, зал/камера, группа.
Кропы не сохраняются на диск (D-016): признаки считаются прямо с исходного кадра.

    python -m qonaq data index            # data/index/roles.csv, data/index/tables.csv + сводка
    python -m qonaq data index --no-mine  # без добора гостей детектором для зала C
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from qonaq.datasets.registry import DATASETS, DatasetSpec, role_specs, table_specs
from qonaq.datasets.yolo import YoloImage, dedupe, iter_images, source_key

log = logging.getLogger("qonaq.datasets")

# Кроп человека считается «нормальным» от этого размера (в пикселях кадра датасета).
MIN_W, MIN_H = 24, 48
MINE_IOU = 0.3      # гость зала C = человек детектора с IoU < 0.3 со всеми Waiter
MINE_CONF = 0.5     # и уверенностью детектора не ниже этой


@dataclass
class ObjRow:
    dataset: str
    venue: str        # роли: A/B/C/D; столы: имя камеры (cam1…, photo_kN)
    group: str        # единица честного разбиения внутри venue (блок времени / дата / камера)
    source: str
    split_orig: str   # родной split Roboflow — НЕ использовать для разбиений
    image: str        # путь относительно корня проекта
    img_w: int
    img_h: int
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    small: int        # 1 — меньше MIN_W×MIN_H (отдельная корзина)
    origin: str       # gt | mined


def is_small(w: int, h: int) -> int:
    """«Мелкий» кроп: площадь меньше 24×48 или короткая сторона < 16 px. Площадь, а не 24×48 по сторонам —
    сидящие люди сверху почти квадратные (D-016)."""
    return int(w * h < MIN_W * MIN_H or min(w, h) < 16)


def _num(pattern: str, s: str) -> int | None:
    m = re.search(pattern, s)
    return int(m.group(1)) if m else None


def role_group(spec: DatasetSpec, source: str) -> str:
    """Блок времени внутри зала — для «верхней границы» (разбиение внутри одного зала)."""
    if spec.venue == "A":
        n = _num(r"image_(\d+)", source)
        return f"A{(n or 0) // 500:02d}"
    if spec.venue == "B":
        n = _num(r"videos_(\d+)", source)
        return f"B{(n or 0) // 100:02d}"
    if spec.venue == "C":
        n = _num(r"ramada_res_(\d+)", source)
        return f"C{(n or 0) // 50:02d}"
    if spec.venue == "D":
        m = re.search(r"IPC_main_(\d{8})", source)
        return f"D{m.group(1) if m else 'x'}"
    return spec.venue


def table_camera(source: str) -> str:
    m = re.match(r"(cam\d+)_", source)
    if m:
        return m.group(1)
    if source.startswith("photo_"):
        return "photo"  # уточняется кластеризацией (cluster_photo_cameras)
    return "other"


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _rows_for_image(spec: DatasetSpec, im: YoloImage, root: Path, venue: str, group: str,
                    size: tuple[int, int]) -> list[ObjRow]:
    w, h = size
    rows = []
    for b in im.boxes:
        lab = spec.class_map.get(b.cls_name)
        if lab is None:
            continue
        x1, y1, x2, y2 = int(b.x1 * w), int(b.y1 * h), int(round(b.x2 * w)), int(round(b.y2 * h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        small = is_small(x2 - x1, y2 - y1) if spec.kind == "roles" else 0
        rows.append(ObjRow(spec.slug, venue, group, im.source, im.split, _rel(im.path, root), w, h, lab,
                           x1, y1, x2, y2, small, "gt"))
    return rows


def _image_size(p: Path) -> tuple[int, int]:
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise OSError(f"не читается {p}")
    return im.shape[1], im.shape[0]


def mine_guests(images: list[YoloImage], spec: DatasetSpec, root: Path, sizes: dict[str, tuple[int, int]],
                cache: Path, model: str = "yolox_m") -> list[ObjRow]:
    """Зал C: гостей нет в разметке — берём людей детектора, не пересекающихся с Waiter (IoU < MINE_IOU)."""
    cached: dict[str, list[list[float]]] = {}
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    det = None
    rows: list[ObjRow] = []
    for i, im in enumerate(images):
        key = im.source
        if key not in cached:
            if det is None:
                from qonaq.detect.onnx_yolox import OnnxYoloxDetector
                det = OnnxYoloxDetector(model=model, conf=0.3, tiles=1)
            frame = cv2.imread(str(im.path))
            cached[key] = [[d.x1, d.y1, d.x2, d.y2, d.conf] for d in det.detect([frame])[0] if d.label == "person"]
            if i % 50 == 0:
                log.info("добор гостей C: %d/%d", i, len(images))
                cache.write_text(json.dumps(cached), encoding="utf-8")
        w, h = sizes[im.source]
        waiters = [(b.x1 * w, b.y1 * h, b.x2 * w, b.y2 * h) for b in im.boxes if b.cls_name == "Waiter"]
        for x1, y1, x2, y2, conf in cached[key]:
            if conf < MINE_CONF:
                continue
            if any(_iou((x1, y1, x2, y2), wb) >= MINE_IOU for wb in waiters):
                continue
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            small = is_small(x2 - x1, y2 - y1)
            rows.append(ObjRow(spec.slug, spec.venue, role_group(spec, im.source), im.source, im.split,
                               _rel(im.path, root), w, h, "guest", x1, y1, x2, y2, small, "mined"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cached), encoding="utf-8")
    return rows


def build(data_root: Path, project_root: Path, out_dir: Path, mine: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"roles": {}, "tables": {}}
    role_rows: list[ObjRow] = []
    for spec in role_specs():
        ds = data_root / spec.slug
        if not (ds / "data.yaml").exists():
            log.warning("нет датасета %s — пропуск (python scripts/fetch_datasets.py)", spec.slug)
            continue
        all_imgs = list(iter_images(ds))
        imgs = dedupe(all_imgs)
        sizes = {im.source: _image_size(im.path) for im in imgs}
        for im in imgs:
            role_rows += _rows_for_image(spec, im, project_root, spec.venue, role_group(spec, im.source),
                                         sizes[im.source])
        if mine and spec.venue == "C":
            role_rows += mine_guests(imgs, spec, project_root, sizes, out_dir / "cache" / "ramada_person_dets.json")
        summary["roles"][spec.venue] = {"dataset": spec.slug, "images_total": len(all_imgs),
                                        "sources_unique": len(imgs)}
    _write(out_dir / "roles.csv", role_rows)

    table_rows: list[ObjRow] = []
    for spec in table_specs():
        ds = data_root / spec.slug
        if not (ds / "data.yaml").exists():
            log.warning("нет датасета %s — пропуск", spec.slug)
            continue
        all_imgs = list(iter_images(ds))
        imgs = dedupe(all_imgs)
        for im in imgs:
            cam = table_camera(im.source)
            if spec.slug == "dirty_table" and cam.startswith("cam"):
                continue  # те же кадры и та же разметка, что в clean_table (632 общих исходника) — не дублируем
            if spec.slug == "table_detection":
                cam = "ext"
            table_rows += _rows_for_image(spec, im, project_root, cam, cam, _image_size(im.path))
        summary["tables"][spec.slug] = {"images_total": len(all_imgs), "sources_unique": len(imgs)}
    _write(out_dir / "tables.csv", table_rows)

    rc: dict = defaultdict(Counter)
    for r in role_rows:
        rc[r.venue][f"{r.label}{'_small' if r.small else ''}{'_mined' if r.origin == 'mined' else ''}"] += 1
    for v, c in rc.items():
        summary["roles"].setdefault(v, {})["objects"] = dict(sorted(c.items()))
    tc: dict = defaultdict(Counter)
    for r in table_rows:
        tc[f"{r.dataset}/{r.venue}"][r.label] += 1
    summary["tables_by_camera"] = {k: dict(sorted(v.items())) for k, v in sorted(tc.items())}
    # Пересечение исходников между двумя наборами столов (одни и те же кадры cam* в обоих)
    src = defaultdict(set)
    for r in table_rows:
        src[r.dataset].add(r.source)
    if "dirty_table" in src and "clean_table" in src:
        summary["tables"]["shared_sources_dirty_clean"] = len(src["dirty_table"] & src["clean_table"])
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def _write(path: Path, rows: list[ObjRow]) -> None:
    fields = list(ObjRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def load_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("img_w", "img_h", "x1", "y1", "x2", "y2", "small"):
            r[k] = int(r[k])
    return rows


__all__ = ["DATASETS", "build", "load_rows", "source_key"]
