"""Калибровка формы персонала без GUI (TASKS M4).

Шаг 1: `roles calibrate <video>` — прогон видео, в runs/calib/<имя>/ сохраняются кадры с id треков
        (лица размыты) и tracks.json с гистограммами торса каждого трека.
Шаг 2: посмотреть кадры, выписать id сотрудников и запустить
        `roles calibrate <video> --staff-ids 3,17,21` -> configs/staff_profile.yaml.
Авто-режим `--auto`: в профиль попадают треки, побывавшие у >= 3 столов («многостольные»).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from qonaq.config import SiteConfig, site_for_file
from qonaq.pipeline import FrameResult, Pipeline
from qonaq.render.annotate import render_tracks
from qonaq.roles.uniform import UniformProfile, torso_hist

log = logging.getLogger("qonaq.calib")


def collect_tracks(video: str | Path, site: SiteConfig | None = None, out_dir: str | Path | None = None,
                   every_s: float = 2.0, detector=None, max_seconds: float | None = None) -> Path:
    """Прогон видео: кадры с id + tracks.json. Возвращает папку."""
    video = Path(video)
    site = site_for_file(video, site)
    out = Path(out_dir) if out_dir else Path(site.runs_dir) / "calib" / video.stem
    out.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(site, detector=detector, max_seconds=max_seconds)
    cam = next(iter(pipe.cameras.values()))
    acc: dict[int, dict] = {}
    state = {"next": 0.0}

    def on_result(r: FrameResult) -> None:
        for p in r.people:
            h = torso_hist(r.frame, p.xyxy)
            a = acc.setdefault(p.track_id, {"hist": np.zeros(0), "n": 0, "first": r.ts, "tables": set()})
            a["last"] = r.ts
            a["tables"].update(z for z in p.zones if z.startswith("T"))
            if h is not None:
                a["hist"] = h if a["n"] == 0 else a["hist"] + h
                a["n"] += 1
        if r.ts >= state["next"]:
            state["next"] = r.ts + every_s
            img = render_tracks(r, cam.zones, blur=True)
            cv2.imwrite(str(out / f"frame_{int(r.ts):05d}s.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

    pipe.run(on_result=on_result)
    tracks = {
        str(tid): {"frames": a["n"], "first_s": round(a["first"], 1), "last_s": round(a.get("last", a["first"]), 1),
                   "tables": sorted(a["tables"]), "hist": [round(float(v), 5) for v in (a["hist"] / a["n"])]}
        for tid, a in acc.items() if a["n"] >= 3
    }
    (out / "tracks.json").write_text(json.dumps(tracks, ensure_ascii=False), encoding="utf-8")
    log.info("калибровка: %d треков, кадры в %s", len(tracks), out)
    return out


def build_profile(calib_dir: str | Path, staff_ids: list[int] | None = None, auto: bool = False,
                  min_tables: int = 3, threshold: float = 0.55) -> tuple[UniformProfile, list[int]]:
    tracks = json.loads((Path(calib_dir) / "tracks.json").read_text(encoding="utf-8"))
    ids = list(staff_ids or [])
    if auto:
        ids += [int(t) for t, v in tracks.items() if len(v["tables"]) >= min_tables and int(t) not in ids]
    missing = [i for i in ids if str(i) not in tracks]
    if missing:
        raise ValueError(f"нет таких id треков в калибровочном прогоне: {missing}")
    if not ids:
        raise ValueError("не выбран ни один сотрудник (укажите --staff-ids или используйте --auto)")
    refs = [np.array(tracks[str(i)]["hist"], dtype=np.float32) for i in ids]
    note = f"калибровка по трекам {ids}" + (" (авто: многостольные треки)" if auto else "")
    return UniformProfile(refs, threshold, note), ids
