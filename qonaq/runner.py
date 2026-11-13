"""Сценарии поверх пайплайна: прогон файла с аннотированным видео, демо-ролик."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2

from qonaq.config import SiteConfig, site_for_file
from qonaq.events.store import EventStore
from qonaq.pipeline import FrameResult, Pipeline, PipelineStats
from qonaq.render.annotate import DemoRenderer, render_tracks
from qonaq.render.sink import VideoSink

log = logging.getLogger("qonaq.run")


def run_dir_for(site: SiteConfig, name: str, out: str | Path | None = None) -> Path:
    d = Path(out) if out else Path(site.runs_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_run_meta(run_dir: Path, site: SiteConfig, pipe: Pipeline, stats: PipelineStats, extra: dict) -> None:
    meta = {
        "site_id": site.site_id, "name": site.name, "profile": site.profile,
        "cameras": [c.id for c in site.cameras],
        "detector": getattr(pipe.detector, "name", "?"), "device": getattr(pipe.detector, "device", "?"),
        "process_fps": site.process_fps, "frames": stats.frames, "wall_s": round(stats.wall_s, 1),
        "scene_s": round(stats.scene_s, 1), "fps_achieved": round(stats.fps, 2),
        "realtime_factor": round(stats.realtime_factor, 2),
        "detect_ms_per_frame": round(1000 * stats.detect_s / max(1, stats.frames), 1),
        "zones": {cid: [z.id for z in cam.zones.zones] for cid, cam in pipe.cameras.items()},
        **extra,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def process_file(video: str | Path, site: SiteConfig | None = None, out: str | Path | None = None,
                 mode: str = "tracks", blur: bool = True, speed: float = 1.0, max_seconds: float | None = None,
                 detector=None, write_video: bool = True, snapshots: int = 0, pos=None,
                 zones_path: str | Path | None = None, loops: int = 1) -> Path:
    """Прогон видеофайла: события в events.sqlite + аннотированное видео.

    mode='tracks' -> annotated.mp4 (bbox, id, роли); mode='demo' -> demo.mp4 (полигоны + панель).
    """
    video = Path(video)
    base_given = site is not None and bool(site.cameras)
    if not base_given:
        keep = site
        site = site_for_file(video, site)
        if keep is not None:  # пороги/детектор из --site без камер тоже учитываются
            site.site_id = site.name = video.stem
    cam_id = site.cameras[0].id
    if zones_path:
        site.cameras[0].zones = str(zones_path)
    if loops > 1:
        from qonaq.sources.video import probe

        site.cameras[0].loop = True
        dur = probe(video).duration_s or 60.0
        max_seconds = min(max_seconds, dur * loops) if max_seconds else dur * loops - 0.05
    run_dir = run_dir_for(site, cam_id, out)
    db = run_dir / "events.sqlite"
    if db.exists():
        db.unlink()
    store = EventStore(db)
    pipe = Pipeline(site, detector=detector, store=store, max_seconds=max_seconds, pos=pos)
    cam = pipe.cameras[cam_id]
    name = "demo.mp4" if mode == "demo" else "annotated.mp4"
    sink = VideoSink(run_dir / name, fps=site.process_fps * speed) if write_video else None
    demo = None
    snap_dir = run_dir / "frames"
    snap_every = None
    state = {"n": 0}

    from qonaq.health.monitor import HealthMonitor

    health = HealthMonitor()

    def on_result(r: FrameResult) -> None:
        nonlocal demo, snap_every
        health.observe_result(r)
        if mode == "demo":
            if demo is None:
                demo = DemoRenderer(cam.zones, site, blur=blur)
            img = demo.render(r)
        else:
            img = render_tracks(r, cam.zones, blur=blur, top_fraction=site.privacy.blur_top_fraction)
        if sink is not None:
            sink.write(img)
        if snapshots:
            if snap_every is None:
                total = pipe.sources[cam_id].info.duration_s or 60.0
                if max_seconds:
                    total = min(total, max_seconds)
                snap_every = max(1, int(total * site.process_fps / snapshots))
            if state["n"] % snap_every == 0:
                snap_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(snap_dir / f"{mode}_{int(r.ts):04d}s.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        state["n"] += 1

    stats = pipe.run(on_result=on_result, progress_every_s=20)
    if sink is not None:
        sink.close()
    store.set_meta(time_base="video", site_id=site.site_id, profile=site.profile, scene_s=stats.scene_s,
                   tables=sum(len(c.zones.by_type("table")) for c in pipe.cameras.values()))
    _write_run_meta(run_dir, site, pipe, stats, {"video": video.name, "mode": mode, "blur": blur,
                                                "events": len(store.events())})
    import json as _json

    (run_dir / "health.json").write_text(_json.dumps(health.status(live=False), ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    store.close()
    log.info("готово: %s — %d кадров за %.0f с (%.1f кадр/с, x%.2f от реального времени)", run_dir,
             stats.frames, stats.wall_s, stats.fps, stats.realtime_factor)
    return run_dir
