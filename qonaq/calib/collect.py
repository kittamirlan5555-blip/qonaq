"""`calibrate collect`: прогон записи зала → кропы людей (с блюром лица) + признаки + авторазметка по поведению.

Авторазметка трека (только уверенные случаи, остальное — на ручную проверку):
- гость: сидел в зоне стола не меньше `guest_min_s` (официантам сидеть с гостями запрещено);
- персонал: стоя побывал у ≥ `staff_min_tables` разных столов за окно ролей (обход зала).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from qonaq.calib.store import calib_dir
from qonaq.config import SiteConfig

log = logging.getLogger("qonaq.calib")

THUMB_H = 160


def blurred_thumb(frame: np.ndarray, xyxy, margin: float = 0.08) -> np.ndarray | None:
    """Кроп человека для страницы проверки: верх (лицо) пикселизован, одежда видна."""
    from qonaq.render.blur import blur_people
    from qonaq.roles.encoder import crop_box

    c = crop_box(frame, xyxy, margin)
    if c is None:
        return None
    c = c.copy()
    h, w = c.shape[:2]
    blur_people(c, [(0, int(margin * h / (1 + 2 * margin)), w, h)], top_fraction=0.38)
    k = THUMB_H / h
    return cv2.resize(c, (max(8, int(w * k)), THUMB_H), interpolation=cv2.INTER_AREA)


def auto_label(t: dict, guest_min_s: float, staff_min_tables: int) -> str | None:
    if t["seated_dwell_s"] >= guest_min_s and t["standing_share"] < 0.5:
        return "guest"
    if t["max_tables"] >= staff_min_tables and t["standing_share"] >= 0.6:
        return "staff"
    return None


def collect(video: str | Path, site: SiteConfig, site_id: str, encoder: str = "mnv2", every_s: float = 2.0,
            max_per_track: int = 30, guest_min_s: float = 300.0, staff_min_tables: int = 3,
            max_seconds: float | None = None, root: Path | None = None) -> dict:
    from qonaq.events.store import EventStore
    from qonaq.pipeline import Pipeline
    from qonaq.roles.encoder import OnnxEncoder, crop_box
    from qonaq.roles.features import hsv_features
    from qonaq.runner import site_for_file

    video = Path(video)
    if not site.cameras:
        site = site_for_file(video, site)
    out = calib_dir(site_id, root)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    enc = OnnxEncoder(encoder)
    store = EventStore(out / "events.sqlite")
    pipe = Pipeline(site, store=store, max_seconds=max_seconds)
    cam_id = site.cameras[0].id
    cam = pipe.cameras[cam_id]
    last: dict[int, float] = {}
    feats, meta = [], []
    tracks: dict[int, dict] = defaultdict(lambda: {"n": 0, "seated_dwell_s": 0.0, "standing": 0, "obs": 0,
                                                   "max_tables": 0, "first_ts": None, "last_ts": 0.0,
                                                   "thumbs": []})

    def on_result(r) -> None:
        due, boxes = [], []
        for p in r.people:
            tr = tracks[p.track_id]
            tr["obs"] += 1
            tr["standing"] += int(p.standing)
            tr["first_ts"] = r.ts if tr["first_ts"] is None else tr["first_ts"]
            tr["last_ts"] = r.ts
            if p.zones and not p.standing:
                tr["seated_dwell_s"] = max(tr["seated_dwell_s"], p.dwell_s)
            tr["max_tables"] = max(tr["max_tables"], cam.roles.tables_visited(p.track_id, r.ts))
            if r.ts - last.get(p.track_id, -1e9) >= every_s and tr["n"] < max_per_track:
                if crop_box(r.frame, p.xyxy) is not None:
                    due.append(p)
                    boxes.append(p.xyxy)
        if not due:
            return
        crops = [crop_box(r.frame, b) for b in boxes]
        emb = enc.embed_crops(crops)
        for p, e in zip(due, emb, strict=True):
            tr = tracks[p.track_id]
            last[p.track_id] = r.ts
            name = f"{p.track_id}_{int(r.ts * 10):07d}.jpg"
            th = blurred_thumb(r.frame, p.xyxy)
            if th is not None:
                cv2.imwrite(str(out / "crops" / name), th, [cv2.IMWRITE_JPEG_QUALITY, 85])
                tr["thumbs"].append(name)
            feats.append(np.concatenate([e, hsv_features(r.frame, p.xyxy)]))
            x1, y1, x2, y2 = p.xyxy
            meta.append((p.track_id, r.ts, x2 - x1, y2 - y1, int(p.standing), p.role))
            tr["n"] += 1

    stats = pipe.run(on_result=on_result, progress_every_s=30)
    store.close()
    tr_out = []
    for tid, t in sorted(tracks.items()):
        if t["n"] == 0:
            continue
        t = dict(t)
        t["standing_share"] = round(t.pop("standing") / max(1, t.pop("obs")), 2)
        t["key"] = f"t{tid}"
        t["track_id"] = tid
        t["auto"] = auto_label(t, guest_min_s, staff_min_tables)
        tr_out.append(t)
    X = np.stack(feats).astype(np.float32) if feats else np.zeros((0, enc.dim + 256), np.float32)
    m = np.array(meta, dtype=object) if meta else np.zeros((0, 6), dtype=object)
    np.savez_compressed(out / "samples.npz", X=X,
                        track=np.array([int(v) for v in m[:, 0]], np.int64) if len(m) else np.zeros(0, np.int64),
                        ts=np.array([float(v) for v in m[:, 1]], np.float32) if len(m) else np.zeros(0, np.float32),
                        w=np.array([float(v) for v in m[:, 2]], np.float32) if len(m) else np.zeros(0, np.float32),
                        h=np.array([float(v) for v in m[:, 3]], np.float32) if len(m) else np.zeros(0, np.float32),
                        encoder=np.array(encoder))
    (out / "tracks.json").write_text(json.dumps(tr_out, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"video": str(video), "scene_s": round(stats.scene_s, 1), "samples": int(len(X)),
               "tracks": len(tr_out), "auto_guest": sum(t["auto"] == "guest" for t in tr_out),
               "auto_staff": sum(t["auto"] == "staff" for t in tr_out),
               "to_review": sum(t["auto"] is None for t in tr_out), "encoder": encoder,
               "rules": {"guest_min_s": guest_min_s, "staff_min_tables": staff_min_tables}}
    (out / "collect.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("калибровка: %s", summary)
    return summary
