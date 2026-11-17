"""Разметка зон мест по кадру с людьми (docs/ZONES_GUIDE.md, D-046): траектории тех, кто садился, и к какому столу
относится каждая группа бёдер.

    python scripts/zone_hips.py data/videos/x.mp4 --site configs/sites/x.yaml --start 0 --seconds 300 --fps 2 \
        --zones configs/zones/x.yaml --out runs/zones/x_hips.jpg

Детектор + трекер + поза на `--fps` кадрах в секунду. Для каждого трека, который сидел (поза «сидит») не меньше
`--sit-s` секунд:
- линия — путь бёдер с момента появления до места, где сел (откуда подошёл);
- кружок с номером — место посадки (медиана бёдер, пока сидел);
- подпись — стол из `--zones`, в зону мест которого попало место, или «?» (не попало ни в одну).

Точки всех остальных: зелёный — сидит, синий — стоит, красный — нагнулся, серый — не понять. Кадр — с блюром лиц.
Рядом с картинкой — `.txt` со списком сидевших: №, место, откуда пришёл, стол. Группа «?» рядом со столом — повод
поправить зону; группа, пришедшая с другой стороны и сидящая «у края», — сосед из другого ряда.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qonaq.config import load_site  # noqa: E402
from qonaq.detect.base import PERSON, create_detector  # noqa: E402
from qonaq.detect.pose import hips_point, posture  # noqa: E402
from qonaq.pipeline import get_pose_estimator  # noqa: E402
from qonaq.render.blur import blur_people  # noqa: E402
from qonaq.track.tracker import PersonTracker  # noqa: E402

COL = {"sit": (0, 220, 0), "stand": (255, 140, 0), "bend": (0, 0, 255), None: (170, 170, 170)}
PALETTE = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255), (147, 20, 255), (50, 205, 50), (255, 105, 180)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--site", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--sit-s", type=float, default=10.0, help="сколько секунд сидел — «сидевший»")
    ap.add_argument("--zones")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    site = load_site(a.site)
    det = create_detector(site.detector)
    pe = get_pose_estimator(site.detector.device)
    t_cfg = site.tracker
    trk = PersonTracker(a.fps, t_cfg.track_activation_threshold, t_cfg.lost_track_buffer_s,
                        t_cfg.minimum_matching_threshold, t_cfg.minimum_consecutive_frames, t_cfg.history_s)
    zones = []
    if a.zones:
        import yaml

        zones = yaml.safe_load(Path(a.zones).read_text(encoding="utf-8"))["zones"]
    cap = cv2.VideoCapture(a.video)
    paths: dict[int, list[tuple[float, float, float, str | None]]] = defaultdict(list)
    pts, base = [], None
    step = 1.0 / a.fps
    t = a.start
    while t <= a.start + a.seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        ds = [d for d in det.detect([f])[0] if d.label == PERSON and d.conf >= site.detector.conf]
        tracks = trk.update(ds, t)
        ks = pe.estimate(f, [tr.xyxy for tr in tracks])
        for tr, k in zip(tracks, ks, strict=True):
            h = hips_point(k)
            if h is None:
                continue
            p = posture(k, tr.xyxy)
            paths[tr.id].append((t, h[0], h[1], p))
            pts.append((h[0], h[1], p))
        if base is None or abs(t - (a.start + a.seconds / 2)) < step / 2:
            base = blur_people(f.copy(), [d.xyxy for d in ds], 0.45)
        t += step
    img = base
    h_img, w_img = img.shape[:2]
    for x, y, p in pts:
        cv2.circle(img, (int(x), int(y)), max(2, w_img // 480), COL[p], -1)
    for z in zones:
        poly = np.array(z["polygon"], np.int32)
        cv2.polylines(img, [poly], True, (0, 255, 255), max(2, w_img // 640))
        if z.get("approach"):
            cv2.polylines(img, [np.array(z["approach"], np.int32)], True, (0, 200, 255), 1)
        cv2.putText(img, z["id"], tuple(int(v) for v in poly[0]), cv2.FONT_HERSHEY_SIMPLEX, w_img / 1600,
                    (0, 255, 255), 2)
    lines = []
    n = 0
    for tid, path in paths.items():
        sit = [(x, y) for _, x, y, p in path if p == "sit"]
        if len(sit) / a.fps < a.sit_s:
            continue
        n += 1
        seat = np.median(np.array(sit), axis=0)
        first_sit = next(i for i, (_, _, _, p) in enumerate(path) if p == "sit")
        route = np.array([(x, y) for _, x, y, _ in path[:first_sit + 1]], np.int32)
        where = "?"
        for z in zones:
            if cv2.pointPolygonTest(np.array(z["polygon"], np.float32), (float(seat[0]), float(seat[1])), False) >= 0:
                where = z["id"]
        c = PALETTE[n % len(PALETTE)]
        if len(route) > 1:
            cv2.polylines(img, [route], False, c, 2)
        cv2.circle(img, tuple(int(v) for v in seat), max(8, w_img // 160), c, 3)
        cv2.putText(img, f"{n}:{where}", (int(seat[0]) + 10, int(seat[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    w_img / 1900, c, 2)
        lines.append(f"{n}\tтрек {tid}\tместо ({seat[0]:.0f}, {seat[1]:.0f})\tпришёл из ({route[0][0]}, {route[0][1]})"
                     f"\tсидел {len(sit) / a.fps:.0f} с\tстол {where}")
    for gx in range(0, w_img, 100):
        cv2.line(img, (gx, 0), (gx, h_img), (90, 90, 90), 1)
        cv2.putText(img, str(gx), (gx + 2, h_img - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for gy in range(0, h_img, 100):
        cv2.line(img, (0, gy), (w_img, gy), (90, 90, 90), 1)
        cv2.putText(img, str(gy), (2, gy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(out, len(pts), "сидевших:", n)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
