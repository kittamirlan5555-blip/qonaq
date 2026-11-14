"""S2: recall людей и скорость детектора на кадрах тёмной угловой камеры (ручная разметка в tests/fixtures).

    python scripts/bench_detector.py                       # все варианты, таблица в runs/s2/bench.md
    python scripts/bench_detector.py --variants yolox_s:1 yolox_m:2:roi --res 960

Разметка: tests/fixtures/gt/people_cctv_restaurant_corner.json — рамки людей на кадрах исходного ролика
(1920×1080), номера кадров. Кадры берутся из data/videos/cctv_restaurant_corner.mp4 (scripts/fetch_videos.py).
Вариант: модель:плитки[:roi][:int8]; --res — ширина, до которой кадр уменьшается (960 ≈ субпоток регистратора).
Совпадение — IoU ≥ 0.5 (и отдельно ≥ 0.3: у сидящих людей рамка часто только по видимой части).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qonaq.detect.onnx_yolox import OnnxYoloxDetector  # noqa: E402

GT = ROOT / "tests" / "fixtures" / "gt" / "people_cctv_restaurant_corner.json"
VIDEO = ROOT / "data" / "videos" / "cctv_restaurant_corner.mp4"
DEFAULT = ["yolox_tiny:1", "yolox_s:1", "yolox_s:2", "yolox_s:3", "yolox_s:2:roi", "yolox_s:3:roi",
           "yolox_m:1", "yolox_m:2", "yolox_m:2:roi", "yolox_m:3:roi", "yolox_s:1:int8", "yolox_s:2:roi:int8",
           "nanodet-plus-m-1.5x_416:1", "nanodet-plus-m-1.5x_416:3:roi"]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(gt: list, pred: list, thr: float) -> tuple[int, int]:
    """Жадное сопоставление по IoU: (найдено из разметки, ложных)."""
    used = set()
    tp = 0
    for g in gt:
        best, bj = thr, None
        for j, p in enumerate(pred):
            if j in used:
                continue
            v = iou(g, p)
            if v >= best:
                best, bj = v, j
        if bj is not None:
            used.add(bj)
            tp += 1
    return tp, len(pred) - len(used)


def load_frames(indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(VIDEO))
    out = {}
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            out[i] = f
    cap.release()
    return out


def roi_of(gt: dict) -> tuple[float, float, float, float] | None:
    """ROI как в пайплайне: прямоугольник зон столов этой камеры (+15%)."""
    from qonaq.zones.models import load_zones

    zs = load_zones(ROOT / "configs" / "zones" / "cctv_restaurant_corner.yaml").scaled(1920, 1080)
    pts = [p for z in zs.zones if z.type in ("table", "waiting_area", "service_point") for p in z.polygon]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = 0.15 * (max(xs) - min(xs)), 0.15 * (max(ys) - min(ys)) + 0.05 * 1080
    return max(0.0, min(xs) - mx), max(0.0, min(ys) - my), min(1920.0, max(xs) + mx), min(1080.0, max(ys) + my)


_ZONES = None


def in_zones(x: float, y: float) -> bool:
    """Центр рамки внутри одной из зон столов камеры — «человек за столом» (то, что важно продукту)."""
    global _ZONES
    if _ZONES is None:
        from qonaq.zones.models import load_zones

        zs = load_zones(ROOT / "configs" / "zones" / "cctv_restaurant_corner.yaml").scaled(1920, 1080)
        _ZONES = [np.array(z.polygon, np.float32) for z in zs.zones if z.type == "table"]
    return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in _ZONES)


def model_path(name: str, int8: bool) -> str:
    if not int8:
        return name
    p = ROOT / "data" / "weights" / f"{name}_int8.onnx"
    if not p.exists():
        raise SystemExit(f"нет {p}: python scripts/quantize_detector.py {name}")
    return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=DEFAULT)
    ap.add_argument("--res", type=int, nargs="*", default=[1920, 960], help="ширина входного кадра")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--threads", type=int, default=0, help="потоков onnxruntime (0 — все ядра)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "s2" / "bench.md"))
    args = ap.parse_args(argv)
    gt = json.loads(GT.read_text(encoding="utf-8"))
    frames = load_frames([f["frame"] for f in gt["frames"]])
    base_roi = roi_of(gt)
    n_gt = sum(len(f["boxes"]) for f in gt["frames"])
    rows = []
    for var in args.variants:
        parts = var.split(":")
        name, tiles = parts[0], int(parts[1]) if len(parts) > 1 else 1
        use_roi, int8 = "roi" in parts, "int8" in parts
        if name.startswith("nanodet"):
            from qonaq.detect.onnx_nanodet import OnnxNanoDetDetector

            det = OnnxNanoDetDetector(name, conf=args.conf, tiles=tiles, threads=args.threads or None)
        else:
            det = OnnxYoloxDetector(model_path(name, int8), conf=args.conf, tiles=tiles,
                                    threads=args.threads or None)
        for res in args.res:
            k = res / 1920
            roi = tuple(v * k for v in base_roi) if use_roi else None
            tp5 = tp3 = fp5 = 0
            small_tp = small_n = 0
            zone_tp = zone_n = 0
            times = []
            first = True
            for f in gt["frames"]:
                img = frames[f["frame"]]
                if res != img.shape[1]:
                    img = cv2.resize(img, (res, int(round(img.shape[0] * k))), interpolation=cv2.INTER_AREA)
                if first:  # прогрев
                    det.detect([img], [roi])
                    first = False
                t0 = time.perf_counter()
                d = det.detect([img], [roi])[0]
                times.append(time.perf_counter() - t0)
                pred = [[v / k for v in x.xyxy] for x in d if x.label == "person"]
                a, b = match(f["boxes"], pred, 0.5)
                tp5, fp5 = tp5 + a, fp5 + b
                tp3 += match(f["boxes"], pred, 0.3)[0]
                inz = [g for g in f["boxes"] if in_zones((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)]
                zone_n += len(inz)
                zone_tp += match(inz, pred, 0.3)[0]
                small = [g for g in f["boxes"] if (g[3] - g[1]) < 160]  # дальние/мелкие (< 15% высоты кадра)
                small_n += len(small)
                small_tp += match(small, pred, 0.3)[0]
            ms = 1000 * float(np.median(times))
            row = {"variant": var, "res": res, "recall50": tp5 / n_gt, "recall30": tp3 / n_gt,
                   "precision50": tp5 / max(1, tp5 + fp5), "small_recall30": small_tp / max(1, small_n),
                   "zone_recall30": zone_tp / max(1, zone_n), "zone_n": zone_n,
                   "ms": ms, "fps_1core_equiv": None}
            rows.append(row)
            print(f"{var:22s} {res:5d}px  recall@.5 {row['recall50']:.2f}  @.3 {row['recall30']:.2f}  "
                  f"мелкие@.3 {row['small_recall30']:.2f}  за столами@.3 {row['zone_recall30']:.2f}  "
                  f"precision@.5 {row['precision50']:.2f}  {ms:6.0f} мс/кадр", flush=True)
    lines = [f"Кадров: {len(gt['frames'])}, людей в разметке: {n_gt} (мелких < 160 px: "
             f"{sum(1 for f in gt['frames'] for g in f['boxes'] if g[3] - g[1] < 160)}); conf {args.conf}.", "",
             "| вариант | вход | recall IoU≥0,5 | recall IoU≥0,3 | мелкие, IoU≥0,3 | за столами, IoU≥0,3 | "
             "precision IoU≥0,5 | мс/кадр |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['variant']}` | {r['res']} | {r['recall50']:.2f} | {r['recall30']:.2f} | "
                     f"{r['small_recall30']:.2f} | {r['zone_recall30']:.2f} | {r['precision50']:.2f} | {r['ms']:.0f} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
