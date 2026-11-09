"""LiveService: пайплайн в фоновом потоке + общее состояние для веб-панели.

Цикл обработки ничего не ждёт: события пишет AsyncEventWriter, алерты шлёт AlertDispatcher,
панель читает последний снимок состояния и последний JPEG под замком.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from qonaq.alerts.telegram import AlertDispatcher, sender_from_env
from qonaq.config import SiteConfig
from qonaq.events.store import AsyncEventWriter, EventStore
from qonaq.pipeline import FrameResult, Pipeline
from qonaq.render.annotate import draw_people, draw_zones
from qonaq.render.blur import blur_people
from qonaq.sources.video import parse_source

log = logging.getLogger("qonaq.live")


class LiveService:
    def __init__(self, site: SiteConfig, detector=None, run_dir: str | Path | None = None, sender=None,
                 use_env_sender: bool = True, max_seconds: float | None = None, pos=None,
                 jpeg_width: int = 960):
        self.site = site
        self.run_dir = Path(run_dir) if run_dir else Path(site.runs_dir) / f"live_{site.site_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = EventStore(self.run_dir / "events.sqlite")
        kinds = {parse_source(c.source)[0] for c in site.cameras}
        self.time_base = "video" if kinds == {"file"} else "epoch"
        self.alerts = AlertDispatcher(sender if sender is not None else (sender_from_env() if use_env_sender else None),
                                      site_name=site.name)
        self.writer = AsyncEventWriter(self.store)
        if pos is None and site.pos.adapter != "none":
            from qonaq.pos import create_pos

            pos = create_pos(site.pos)
        self.pipeline = Pipeline(site, detector=detector, store=self.writer, max_seconds=max_seconds, pos=pos)
        self.jpeg_width = jpeg_width
        self._lock = threading.Lock()
        self._jpeg: dict[str, bytes] = {}
        self._state: dict[str, dict[str, Any]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_wall = time.time()
        self.finished = threading.Event()
        self.error: str | None = None
        self._tables = sum(len(c.zones.by_type("table")) for c in self.pipeline.cameras.values())
        self.store.set_meta(time_base=self.time_base, site_id=site.site_id, profile=site.profile,
                            tables=self._tables, started_wall=self.started_wall)

    # ------------------------------------------------------------------ pipeline thread
    def _on_result(self, r: FrameResult) -> None:
        cam = self.pipeline.cameras[r.camera_id]
        titles = {z.id: z.title for z in cam.zones.zones}
        self.alerts.handle(r.events, titles)
        img = r.frame.copy()
        if self.site.privacy.blur_faces:
            blur_people(img, r.blur_boxes, self.site.privacy.blur_top_fraction)
        draw_zones(img, cam.zones, r.zone_states, self.site.alerts.no_waiter_s, show_ignore=False,
                   font=max(14, img.shape[1] // 70))
        draw_people(img, r, show_ids=False, thickness=1)
        if img.shape[1] > self.jpeg_width:
            s = self.jpeg_width / img.shape[1]
            img = cv2.resize(img, (self.jpeg_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        zones = []
        for z in cam.zones.zones:
            if z.type == "ignore":
                continue
            snap = dict(r.zone_states.get(z.id, {}))
            snap.update({"zone_id": z.id, "name": z.name, "type": z.type})
            zones.append(snap)
        state = {"camera_id": r.camera_id, "name": cam.cam.title, "ts": r.ts, "summary": r.summary, "zones": zones,
                 "frame": r.frame_index, "updated_wall": time.time()}
        with self._lock:
            if ok:
                self._jpeg[r.camera_id] = buf.tobytes()
            self._state[r.camera_id] = state

    def _run(self) -> None:
        try:
            self.pipeline.run(on_result=self._on_result, stop=self._stop, progress_every_s=60)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            log.exception("пайплайн остановился с ошибкой")
        finally:
            self.finished.set()

    def start(self) -> LiveService:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        self.writer.close()
        last = max((s["ts"] for s in self._state.values()), default=0.0)
        self.store.set_meta(t_end=last, scene_s=self.pipeline.stats.scene_s)
        self.alerts.close()
        self.store.close()
        import json

        stats = self.pipeline.stats
        (self.run_dir / "run.json").write_text(json.dumps({
            "site_id": self.site.site_id, "name": self.site.name, "profile": self.site.profile, "mode": "live",
            "cameras": [c.id for c in self.site.cameras], "detector": getattr(self.pipeline.detector, "name", "?"),
            "frames": stats.frames, "wall_s": round(stats.wall_s, 1), "fps_achieved": round(stats.fps, 2),
            "zones": {cid: [z.id for z in cam.zones.zones] for cid, cam in self.pipeline.cameras.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ read side (API)
    def jpeg(self, camera_id: str) -> bytes | None:
        with self._lock:
            return self._jpeg.get(camera_id)

    def state(self) -> dict[str, Any]:
        with self._lock:
            cams = [dict(s) for s in self._state.values()]
        stats = self.pipeline.stats
        elapsed = max(time.time() - self.started_wall, 1e-6)
        return {
            "site": {"id": self.site.site_id, "name": self.site.name, "profile": self.site.profile},
            "cameras": cams,
            "camera_ids": [c.id for c in self.site.cameras],
            "alerts": [a.to_dict() for a in list(self.alerts.feed)[-30:]][::-1],
            "telegram": self.alerts.sender is not None,
            "alerts_pending": self.alerts.pending(),
            "time_base": self.time_base,
            "fps": round(stats.frames / elapsed, 2),
            "running": not self.finished.is_set(),
            "error": self.error,
            "server_time": time.time(),
        }

    def events(self, since_id: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.events(since_id=since_id, limit=limit, newest_first=since_id == 0)
