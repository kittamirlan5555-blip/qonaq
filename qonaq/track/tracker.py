"""Трекер людей: обёртка над ByteTrack из supervision (MIT).

Трек хранит историю позиций (нижняя центральная точка bbox) во времени сцены.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from qonaq.detect.base import Detection


@dataclass
class Track:
    id: int
    xyxy: tuple[float, float, float, float]
    conf: float
    first_ts: float
    last_ts: float
    history: deque = field(default_factory=deque)  # (ts, x, y) нижней центральной точки

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return (x1 + x2) / 2, y2

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def age_s(self) -> float:
        return self.last_ts - self.first_ts


class PersonTracker:
    def __init__(self, fps: float = 5.0, activation_threshold: float = 0.3, lost_buffer_s: float = 3.0,
                 matching_threshold: float = 0.8, min_consecutive_frames: int = 2, history_s: float = 30.0):
        import supervision as sv

        self._sv = sv
        self.fps = fps
        self.history_s = history_s
        self._bt = sv.ByteTrack(
            track_activation_threshold=activation_threshold,
            lost_track_buffer=max(1, int(round(lost_buffer_s * fps))),
            minimum_matching_threshold=matching_threshold,
            frame_rate=max(1, int(round(fps))),
            minimum_consecutive_frames=min_consecutive_frames,
        )
        self._tracks: dict[int, Track] = {}

    def update(self, detections: Sequence[Detection], ts: float) -> list[Track]:
        """Принимает детекции людей одного кадра, возвращает активные треки."""
        sv = self._sv
        if detections:
            dets = sv.Detections(
                xyxy=np.array([d.xyxy for d in detections], dtype=np.float32),
                confidence=np.array([d.conf for d in detections], dtype=np.float32),
                class_id=np.zeros(len(detections), dtype=int),
            )
        else:
            dets = sv.Detections.empty()
        tracked = self._bt.update_with_detections(dets)
        active: list[Track] = []
        if tracked.tracker_id is not None:
            for box, conf, tid in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id, strict=True):
                tid = int(tid)
                xyxy = tuple(float(v) for v in box)
                tr = self._tracks.get(tid)
                if tr is None:
                    tr = self._tracks[tid] = Track(tid, xyxy, float(conf), ts, ts)
                tr.xyxy, tr.conf, tr.last_ts = xyxy, float(conf), ts
                bx, by = tr.bottom_center
                tr.history.append((ts, bx, by))
                while tr.history and tr.history[0][0] < ts - self.history_s:
                    tr.history.popleft()
                active.append(tr)
        # забываем давно пропавшие треки
        for tid in [t for t, tr in self._tracks.items() if ts - tr.last_ts > self.history_s]:
            del self._tracks[tid]
        return active
