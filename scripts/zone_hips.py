"""Точки бёдер людей по позе поверх кадра — для разметки зон мест (docs/ZONES_GUIDE.md) и проверки превью.

    python scripts/zone_hips.py data/videos/x.mp4 --site configs/sites/x.yaml --start 0 --seconds 300 --every 10 \
        --zones configs/zones/x.yaml --out runs/zones/x_hips.jpg

Зелёный — сидит, синий — стоит, красный — нагнулся, серый — поза не определилась. Если задан `--zones`, рисует
полигоны: бёдра сидящих за столом — внутри, проход/соседи — снаружи. Кадр в превью — с блюром лиц.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qonaq.config import load_site  # noqa: E402
from qonaq.detect.base import PERSON, create_detector  # noqa: E402
from qonaq.detect.pose import hips_point, posture  # noqa: E402
from qonaq.pipeline import get_pose_estimator  # noqa: E402
from qonaq.render.blur import blur_people  # noqa: E402

COL = {"sit": (0, 220, 0), "stand": (255, 140, 0), "bend": (0, 0, 255), None: (170, 170, 170)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--site", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--zones")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    site = load_site(a.site)
    det = create_detector(site.detector)
    pe = get_pose_estimator(site.detector.device)
    cap = cv2.VideoCapture(a.video)
    pts, base = [], None
    t = a.start
    while t <= a.start + a.seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        ds = [d for d in det.detect([f])[0] if d.label == PERSON and d.conf >= site.detector.conf]
        for d, k in zip(ds, pe.estimate(f, [d.xyxy for d in ds]), strict=True):
            h = hips_point(k)
            if h is not None:
                pts.append((h[0], h[1], posture(k, d.xyxy)))
        if base is None or abs(t - (a.start + a.seconds / 2)) < a.every / 2:
            base = blur_people(f.copy(), [d.xyxy for d in ds], 0.45)
        t += a.every
    img = base
    h, w = img.shape[:2]
    for x, y, p in pts:
        cv2.circle(img, (int(x), int(y)), max(3, w // 320), COL[p], -1)
    if a.zones:
        import yaml

        for z in yaml.safe_load(Path(a.zones).read_text(encoding="utf-8"))["zones"]:
            poly = np.array(z["polygon"], np.int32)
            cv2.polylines(img, [poly], True, (0, 255, 255), max(2, w // 640))
            cv2.putText(img, z["id"], tuple(int(v) for v in poly[0]), cv2.FONT_HERSHEY_SIMPLEX, w / 1600,
                        (0, 255, 255), 2)
    for gx in range(0, w, 100):
        cv2.line(img, (gx, 0), (gx, h), (90, 90, 90), 1)
        cv2.putText(img, str(gx), (gx + 2, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for gy in range(0, h, 100):
        cv2.line(img, (0, gy), (w, gy), (90, 90, 90), 1)
        cv2.putText(img, str(gy), (2, gy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, img)
    np.save(Path(a.out).with_suffix(".npy"),
            np.array([(x, y, {"sit": 0, "stand": 1, "bend": 2}.get(p, 3)) for x, y, p in pts], np.float32))
    print(a.out, len(pts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
