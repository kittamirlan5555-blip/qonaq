"""Цикл обработки (SPEC §9a п.6): камера -> семплер -> общий детектор (батч по камерам)
-> трекер на камеру -> зоны -> роли -> логика -> события.

Детектор один на процесс, остальное — на камеру. Запись событий и панель не блокируют цикл
(AsyncEventWriter, колбэки с результатами кадров).
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qonaq.config import CameraConfig, SiteConfig, resolve_path
from qonaq.detect.base import PERSON, Detection, Detector, create_detector
from qonaq.logic.base import Event, ZoneObservation
from qonaq.logic.engine import Engine, create_engine
from qonaq.roles.tracker import GUEST, STAFF, PersonObs, RoleTracker
from qonaq.roles.uniform import UniformProfile, torso_hist
from qonaq.sources.video import VideoSource
from qonaq.track.tracker import PersonTracker
from qonaq.zones.assign import ZoneAssigner
from qonaq.zones.models import ZoneSet, load_zones

log = logging.getLogger("qonaq.pipeline")


@dataclass
class PersonView:
    track_id: int
    xyxy: tuple[float, float, float, float]
    role: str
    confidence: float
    zones: list[str]
    standing: bool = False
    dwell_s: float = 0.0


@dataclass
class FrameResult:
    camera_id: str
    ts: float
    frame: np.ndarray  # сырой кадр BGR — НЕ сохранять без блюра
    people: list[PersonView]
    blur_boxes: list[tuple[float, float, float, float]]
    objects: list[Detection]  # столы/стулья (для отладки и автопредложения)
    zone_states: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    events: list[Event] = field(default_factory=list)
    frame_index: int = 0


def default_zones_path(camera_id: str) -> Path:
    return resolve_path(Path("configs") / "zones" / f"{camera_id}.yaml")


def load_camera_zones(cam: CameraConfig) -> ZoneSet:
    path = resolve_path(cam.zones) if cam.zones else default_zones_path(cam.id)
    if path.exists():
        zs = load_zones(path)
        zs.camera_id = zs.camera_id or cam.id
        return zs
    if cam.zones:
        log.warning("файл зон не найден: %s — работаю без зон", path)
    return ZoneSet(camera_id=cam.id)


class CameraRuntime:
    """Всё состояние одной камеры: трекер, зоны, роли, логика."""

    def __init__(self, site: SiteConfig, cam: CameraConfig, zones: ZoneSet | None = None, pos=None,
                 uniform: UniformProfile | None = None, role_clf=None):
        self.site = site
        self.role_clf = role_clf
        self._app_last: dict[int, float] = {}  # id трека -> когда последний раз считали внешний вид
        self.cam = cam
        self.fps = cam.process_fps or site.process_fps
        self._zones_src = zones if zones is not None else load_camera_zones(cam)
        self.zones = self._zones_src
        self._scaled_for: tuple[int, int] | None = None
        self._pos = pos
        t = site.tracker
        self.tracker = PersonTracker(self.fps, t.track_activation_threshold, t.lost_track_buffer_s,
                                     t.minimum_matching_threshold, t.minimum_consecutive_frames, t.history_s)
        self.uniform = uniform
        self.roles = RoleTracker(site.roles, site.thresholds.seat_min_s, uniform)
        self.assigner = ZoneAssigner(self.zones)
        self.engine: Engine = create_engine(self.zones, site, pos=pos)
        self.appearance = None
        self.frame_index = 0
        self.last_result: FrameResult | None = None

    def _ensure_scale(self, frame: np.ndarray) -> None:
        """Зоны размечены на кадре одного размера, а поток может прийти другого — пересчитываем один раз."""
        h, w = frame.shape[:2]
        if self._scaled_for == (w, h):
            return
        self._scaled_for = (w, h)
        scaled = self._zones_src.scaled(w, h)
        if scaled is not self.zones:
            self.zones = scaled
            self.assigner = ZoneAssigner(self.zones)
            self.engine = create_engine(self.zones, self.site, pos=self._pos)
        if self.site.thresholds.use_table_appearance and self.site.profile == "restaurant":
            from qonaq.zones.appearance import TableAppearance

            self.appearance = TableAppearance(self.zones, (w, h))

    def _appearance(self, frame: np.ndarray, ts: float, tracks) -> dict[int, float]:
        """p(staff) по внешнему виду — для каждого трека не чаще appearance_every_s, одним батчем."""
        if self.role_clf is None or not tracks:
            return {}
        every = self.site.roles.appearance_every_s
        due = [t for t in tracks if ts - self._app_last.get(t.id, -1e9) >= every]
        if not due:
            return {}
        probs = self.role_clf.staff_proba(frame, [t.xyxy for t in due])
        for t in due:
            self._app_last[t.id] = ts
        if len(self._app_last) > 2000:  # забываем старые id
            live = {t.id for t in tracks}
            self._app_last = {k: v for k, v in self._app_last.items() if k in live}
        return {t.id: float(p) for t, p in zip(due, probs, strict=True)}

    def process(self, frame: np.ndarray, ts: float, detections: list[Detection]) -> FrameResult:
        self._ensure_scale(frame)
        site = self.site
        persons_all = [d for d in detections if d.label == PERSON]
        persons = [d for d in persons_all if d.conf >= site.detector.conf]
        tracks = self.tracker.update(persons, ts)
        blur_boxes = [d.xyxy for d in persons_all] + [t.xyxy for t in tracks]

        # --- зоны
        placed: list[tuple[Any, list[str]]] = []
        for tr in tracks:
            placed.append((tr, self.assigner.assign_one(tr.xyxy)))
        table_ids = {z.id for z in self.zones.by_type("table")}

        # --- роли
        app = self._appearance(frame, ts, tracks)
        obs_people = []
        for tr, zones in placed:
            table = next((z for z in zones if z in table_ids), None)
            hist = torso_hist(frame, tr.xyxy) if self.uniform is not None else None
            obs_people.append(PersonObs(tr.id, tr.xyxy, table, hist, app.get(tr.id)))
        decisions = self.roles.update(ts, obs_people)

        # --- наблюдения по зонам
        per_zone: dict[str, dict[str, list[int]]] = {
            z.id: {"guest": [], "staff": [], "unknown": [], "standing": []}
            for z in self.zones.zones if z.type != "ignore"
        }
        people: list[PersonView] = []
        for tr, zones in placed:
            d = decisions[tr.id]
            people.append(PersonView(tr.id, tr.xyxy, d.role, d.confidence, zones, d.standing, d.dwell_s))
            for zid in zones:
                key = d.role if d.role in (GUEST, STAFF) else "unknown"
                sid = d.stable_id or tr.id  # стабильный id переживает смену id трекером
                per_zone[zid][key].append(sid)
                if d.standing:
                    per_zone[zid]["standing"].append(sid)
        dirty = self.appearance.update(frame, ts, [t.xyxy for t in tracks] + [d.xyxy for d in persons_all],
                                       self.engine.snapshot(ts)) if self.appearance else {}
        observations = []
        for z in self.zones.zones:
            if z.type == "ignore":
                continue
            pz = per_zone[z.id]
            behind = None
            if z.type == "service_point" and z.staff_polygon:
                behind = any(self.assigner.in_staff_area(tr.xyxy, z.id) for tr in tracks)
            observations.append(ZoneObservation(
                ts=ts, zone_id=z.id, guest_tracks=tuple(pz["guest"]), staff_tracks=tuple(pz["staff"]),
                unknown_tracks=tuple(pz["unknown"]), table_dirty=dirty.get(z.id), staff_behind_desk=behind,
                standing_tracks=tuple(pz["standing"]) if site.roles.visit_requires_standing else None,
            ))
        events = self.engine.update(observations)
        for e in events:
            e.camera_id = self.cam.id
        summary = self.engine.summary(ts)
        summary["people"] = len(people)
        res = FrameResult(
            camera_id=self.cam.id, ts=ts, frame=frame, people=people, blur_boxes=blur_boxes,
            objects=[d for d in detections if d.label != PERSON], zone_states=self.engine.snapshot(ts),
            summary=summary, events=events, frame_index=self.frame_index,
        )
        self.frame_index += 1
        self.last_result = res
        return res


@dataclass
class PipelineStats:
    frames: int = 0
    wall_s: float = 0.0
    detect_s: float = 0.0
    scene_s: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def realtime_factor(self) -> float:
        return self.scene_s / self.wall_s if self.wall_s > 0 else 0.0


class Pipeline:
    def __init__(self, site: SiteConfig, detector: Detector | None = None, store=None,
                 zones: dict[str, ZoneSet] | None = None, max_seconds: float | None = None,
                 pos=None, sources: dict[str, VideoSource] | None = None, role_classifier=None):
        if not site.cameras:
            raise ValueError("в конфиге объекта нет ни одной камеры")
        self.site = site
        self.detector = detector or create_detector(site.detector)
        self.store = store
        self.stats = PipelineStats()
        uniform = None
        if site.roles.staff_profile:
            p = resolve_path(site.roles.staff_profile)
            if p.exists():
                uniform = UniformProfile.load(p)
                uniform.threshold = site.roles.uniform_threshold
            else:
                log.warning("профиль формы не найден: %s", p)
        from qonaq.roles.classifier import load_role_classifier

        role_clf = role_classifier if role_classifier is not None else load_role_classifier(site.roles.classifier)
        self.cameras: dict[str, CameraRuntime] = {}
        self.sources: dict[str, VideoSource] = {}
        for cam in site.cameras:
            self.cameras[cam.id] = CameraRuntime(site, cam, (zones or {}).get(cam.id), pos=pos, uniform=uniform,
                                                 role_clf=role_clf)
            if sources and cam.id in sources:
                self.sources[cam.id] = sources[cam.id]
            else:
                self.sources[cam.id] = VideoSource(
                    cam.source, cam.process_fps or site.process_fps, loop=cam.loop, realtime=cam.realtime,
                    max_seconds=max_seconds,
                )
        self._opened = False
        self._active: set[str] = set(self.sources)

    def open(self) -> Pipeline:
        if not self._opened:
            for src in self.sources.values():
                src.open()
            self._opened = True
        return self

    def close(self) -> None:
        for src in self.sources.values():
            src.close()
        self._opened = False

    def step(self) -> list[FrameResult] | None:
        """Один проход по всем камерам. None — все источники закончились."""
        batch: list[tuple[str, np.ndarray, float]] = []
        for cam_id in list(self._active):
            item = self.sources[cam_id].read()
            if item is None:
                self._active.discard(cam_id)
                continue
            batch.append((cam_id, item[0], item[1]))
        if not batch:
            return None
        t0 = time.perf_counter()
        dets = self.detector.detect([b[1] for b in batch])
        self.stats.detect_s += time.perf_counter() - t0
        results = []
        for (cam_id, frame, ts), d in zip(batch, dets, strict=True):
            res = self.cameras[cam_id].process(frame, ts, d)
            results.append(res)
            if self.store is not None and res.events:
                self.store.add(res.events)
        self.stats.frames += len(batch)
        return results

    def run(self, on_result: Callable[[FrameResult], None] | None = None,
            stop: threading.Event | None = None, progress_every_s: float = 0.0) -> PipelineStats:
        self.open()
        t_start = time.perf_counter()
        last_log = t_start
        first_ts: dict[str, float] = {}
        last_ts: dict[str, float] = {}
        try:
            while stop is None or not stop.is_set():
                results = self.step()
                if results is None:
                    break
                for r in results:
                    first_ts.setdefault(r.camera_id, r.ts)
                    last_ts[r.camera_id] = r.ts
                    if on_result is not None:
                        on_result(r)
                now = time.perf_counter()
                if progress_every_s and now - last_log >= progress_every_s:
                    last_log = now
                    log.info("обработано %d кадров, %.1f кадр/с, время сцены %.0f с", self.stats.frames,
                             self.stats.frames / (now - t_start), max(last_ts.values(), default=0.0))
        finally:
            self.stats.wall_s = time.perf_counter() - t_start
            self.stats.scene_s = max((last_ts[c] - first_ts[c] for c in last_ts), default=0.0)
            self.close()
        return self.stats
