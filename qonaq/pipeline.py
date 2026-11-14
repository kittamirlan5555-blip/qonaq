"""Цикл обработки (SPEC §9a п.6): камера -> семплер -> общий детектор (батч по камерам)
-> трекер на камеру -> зоны -> роли -> логика -> события.

Детектор один на процесс, остальное — на камеру. Запись событий и панель не блокируют цикл
(AsyncEventWriter, колбэки с результатами кадров).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
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
        self.table_signals = None
        self.felt = None  # B2: анализ сукна бильярдных столов
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
        if self.zones.by_type("billiard_table"):
            from qonaq.billiard.felt import FeltSet

            self.felt = FeltSet(self.zones, (w, h), self.site.billiard)
        th = self.site.thresholds
        if th.use_table_appearance and self.site.profile == "restaurant":
            from qonaq.zones.appearance import TableAppearance

            self.appearance = TableAppearance(self.zones, (w, h))
        if (th.use_table_classifier or th.use_dish_detection) and self.site.profile == "restaurant":
            from qonaq.zones.tablestate import TableSignals, load_table_classifier

            clf = load_table_classifier(th.table_classifier) if th.use_table_classifier else None
            self.table_signals = TableSignals(self.zones, th, clf)

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
            hist = torso_hist(frame, tr.xyxy)  # дёшево; нужна и форме, и проверке подмены id трекером
            obs_people.append(PersonObs(tr.id, tr.xyxy, table, hist, app.get(tr.id)))
        decisions = self.roles.update(ts, obs_people)

        # --- наблюдения по зонам
        per_zone: dict[str, dict[str, list[int]]] = {
            z.id: {"guest": [], "staff": [], "unknown": [], "standing": [], "gconf": []}
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
                if key == GUEST:
                    per_zone[zid]["gconf"].append(d.confidence)
                if d.standing:
                    per_zone[zid]["standing"].append(sid)
        dirty = self.appearance.update(frame, ts, [t.xyxy for t in tracks] + [d.xyxy for d in persons_all],
                                       self.engine.snapshot(ts)) if self.appearance else {}
        people_boxes = [t.xyxy for t in tracks] + [d.xyxy for d in persons_all]
        tsig = self.table_signals.update(frame, ts, people_boxes, detections) if self.table_signals else {}
        felt = self.felt.update(frame) if self.felt else {}
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
                table_signals=tuple(tsig[z.id]) if z.id in tsig else None,
                guest_confidence=float(np.mean(pz["gconf"])) if pz["gconf"] else None,
                standing_tracks=tuple(pz["standing"]) if site.roles.visit_requires_standing else None,
                felt=felt.get(z.id),
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
class CamStats:
    """Счётчики одной камеры для live-статистики (S1): кадры, задержка «кадр пришёл → обработан»."""

    frames: int = 0
    latency_sum: float = 0.0
    latency_max: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=300))  # (wall, latency)

    def add(self, wall: float, latency: float | None) -> None:
        self.frames += 1
        if latency is not None:
            self.latency_sum += latency
            self.latency_max = max(self.latency_max, latency)
            self.recent.append((wall, latency))

    def window(self, now: float, window_s: float) -> dict[str, float]:
        lat = [x[1] for x in tuple(self.recent) if x[0] >= now - window_s]  # снимок: пишет другой поток
        lat_sorted = sorted(lat)
        return {
            "fps": round(len(lat) / window_s, 2),
            "latency_ms_avg": round(1000 * sum(lat) / len(lat), 1) if lat else None,
            "latency_ms_p95": round(1000 * lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 1) if lat else None,
        }


@dataclass
class PipelineStats:
    frames: int = 0
    wall_s: float = 0.0
    detect_s: float = 0.0
    scene_s: float = 0.0
    cams: dict[str, CamStats] = field(default_factory=dict)

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
        if site.thresholds.use_dish_detection:
            site.detector.dishes = True  # сигнал «посуда на столе» требует классов посуды у детектора
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
        self._rois: dict[tuple, tuple | None] = {}
        self._det_count: dict[str, int] = {}
        self._last_dets: dict[str, list[Detection]] = {}

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
        streams = False
        for cam_id in list(self._active):
            src = self.sources[cam_id]
            if getattr(src, "kind", "file") != "file":
                # Потоки — неблокирующе: пропавшая камера не останавливает остальные (S1).
                streams = True
                item = src.poll()
                if item is not None:
                    batch.append((cam_id, item[0], item[1]))
                continue
            item = src.read()
            if item is None:
                self._active.discard(cam_id)
                continue
            batch.append((cam_id, item[0], item[1]))
        if not batch:
            if streams:
                time.sleep(0.01)
                return []
            return None
        dets = self._detect(batch)
        results = []
        for (cam_id, frame, ts), d in zip(batch, dets, strict=True):
            res = self.cameras[cam_id].process(frame, ts, d)
            results.append(res)
            if self.store is not None and res.events:
                self.store.add(res.events)
            now = time.time()
            # Задержка имеет смысл только для потоков: там ts — время получения кадра (wall clock).
            lat = now - ts if getattr(self.sources[cam_id], "kind", "file") != "file" else None
            self.stats.cams.setdefault(cam_id, CamStats()).add(now, lat)
        self.stats.frames += len(batch)
        return results

    def _roi(self, cam_id: str, frame: np.ndarray) -> tuple[float, float, float, float] | None:
        """S2: прямоугольник вокруг зон столов (+15%) — плитки детектора только там."""
        if not self.site.detector.roi_tiles:
            return None
        h, w = frame.shape[:2]
        key = (cam_id, w, h)
        if key not in self._rois:
            zs = self.cameras[cam_id].zones.scaled(w, h)
            pts = [pt for z in zs.zones if z.type in ("table", "waiting_area", "service_point") for pt in z.polygon]
            if not pts:
                self._rois[key] = None
            else:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                mx, my = 0.15 * (max(xs) - min(xs)), 0.15 * (max(ys) - min(ys)) + 0.05 * h
                self._rois[key] = (max(0.0, min(xs) - mx), max(0.0, min(ys) - my), min(float(w), max(xs) + mx),
                                   min(float(h), max(ys) + my))
        return self._rois[key]

    def _detect(self, batch: list[tuple[str, np.ndarray, float]]) -> list[list[Detection]]:
        """Детектор раз в detect_every кадров камеры (S2); между ними — последние детекции этой камеры."""
        every = self.site.detector.detect_every
        need = [i for i, (cam_id, _, _) in enumerate(batch)
                if every <= 1 or self._det_count.get(cam_id, 0) % every == 0 or cam_id not in self._last_dets]
        for cam_id, _, _ in batch:
            self._det_count[cam_id] = self._det_count.get(cam_id, 0) + 1
        out: list[list[Detection] | None] = [None] * len(batch)
        if need:
            frames = [batch[i][1] for i in need]
            t0 = time.perf_counter()
            if self.site.detector.roi_tiles and getattr(self.detector, "detect", None):
                rois = [self._roi(batch[i][0], batch[i][1]) for i in need]
                try:
                    fresh = self.detector.detect(frames, rois)
                except TypeError:  # детектор без поддержки roi
                    fresh = self.detector.detect(frames)
            else:
                fresh = self.detector.detect(frames)
            self.stats.detect_s += time.perf_counter() - t0
            for i, d in zip(need, fresh, strict=True):
                out[i] = d
                self._last_dets[batch[i][0]] = d
        for i, (cam_id, _, _) in enumerate(batch):
            if out[i] is None:
                out[i] = list(self._last_dets.get(cam_id, []))
        return out  # type: ignore[return-value]

    def finish(self) -> list[Event]:
        """Конец записи / штатная остановка: движки, у которых есть finish (billiard), закрывают открытые сессии."""
        out: list[Event] = []
        for cam_id, cam in self.cameras.items():
            fin = getattr(cam.engine, "finish", None)
            if fin is None or cam.last_result is None:
                continue
            evs = fin(cam.last_result.ts)
            for e in evs:
                e.camera_id = cam_id
            out += evs
        if out and self.store is not None:
            self.store.add(out)
        return out

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
                    scene = max((last_ts[c] - first_ts[c] for c in last_ts), default=0.0)
                    log.info("обработано %d кадров, %.1f кадр/с, сцена %.0f с", self.stats.frames,
                             self.stats.frames / (now - t_start), scene)
        finally:
            self.stats.wall_s = time.perf_counter() - t_start
            self.stats.scene_s = max((last_ts[c] - first_ts[c] for c in last_ts), default=0.0)
            self.close()
        return self.stats
