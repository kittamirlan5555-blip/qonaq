"""Запись-повтор: детекции видео + гистограммы торса, без кадров (регрессионные тесты на реальной записи).

Зачем: логику и роли нужно проверять на настоящей записи (постановка 15.11), а само видео в git класть нельзя —
лица, 1,5 ГБ. Повтор хранит по каждому обработанному кадру только время, размер кадра и людей: рамка,
уверенность и HSV-гистограмма торса 8×4×4 (её нельзя восстановить без кадра, а по ней роли проверяют подмену id
трекером). Изображений и вырезов нет — по гистограмме человека не узнать.

    python -m qonaq replay record data/videos/X.mp4 --site configs/sites/staged.yaml --out tests/fixtures/x.jsonl.gz
    python -m qonaq replay run tests/fixtures/replay/x.jsonl.gz --site configs/sites/staged.yaml --zones ... --gt ...

Прогон повтора даёт те же события, что прогон видео тем же детектором с той же частотой кадров, кроме сигналов,
которым нужен сам кадр (эталон пустого стола, классификатор стола/внешности) — в повторе они выключены.
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qonaq.detect.base import PERSON, Detection

log = logging.getLogger("qonaq.replay")

FORMAT = "qonaq-replay-1"
_Q = 255.0  # гистограмма хранится как uint8 (доля от максимального бина), при чтении нормируется обратно


def _pack_hist(h: np.ndarray | None) -> str | None:
    if h is None:
        return None
    m = float(np.max(h))
    if m <= 0:
        return None
    q = np.round(np.asarray(h, np.float32) / m * _Q).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode("ascii")


def _unpack_hist(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    q = np.frombuffer(base64.b64decode(s), np.uint8).astype(np.float32)
    total = float(q.sum())
    return q / total if total > 0 else None


def record(video: str | Path, site, out: str | Path, detector=None, max_seconds: float | None = None) -> dict:
    """Прогон видео детектором с частотой кадров объекта → файл повтора (gzip JSONL)."""
    from qonaq.detect.base import create_detector
    from qonaq.roles.uniform import torso_hist
    from qonaq.sources.video import VideoSource

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    det = detector or create_detector(site.detector)
    fps = site.resolve_fps(getattr(det, "device", None))
    min_conf = site.detector.conf
    n = n_people = 0
    with VideoSource(str(video), fps, realtime=False, max_seconds=max_seconds) as src, \
            gzip.open(out, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"format": FORMAT, "video": Path(video).name, "process_fps": fps,
                            "detector": getattr(det, "name", "?"), "min_conf": min_conf}) + "\n")
        for frame, ts in src:
            h, w = frame.shape[:2]
            people = []
            for d in det.detect([frame])[0]:
                if d.label != PERSON or d.conf < min_conf:
                    continue
                people.append([round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1), round(d.conf, 3),
                               _pack_hist(torso_hist(frame, d.xyxy))])
            f.write(json.dumps({"t": round(ts, 3), "wh": [w, h], "p": people}) + "\n")
            n += 1
            n_people += len(people)
    info = {"frames": n, "people": n_people, "path": str(out), "bytes": out.stat().st_size}
    log.info("повтор записан: %s", info)
    return info


class Replay:
    """Читает файл повтора; отдаёт источник кадров и детектор для Pipeline(sources=…, detector=…)."""

    def __init__(self, path: str | Path, start_s: float = 0.0, end_s: float | None = None):
        self.path = Path(path)
        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            self.header: dict[str, Any] = json.loads(f.readline())
            if self.header.get("format") != FORMAT:
                raise ValueError(f"{path}: не файл повтора ({self.header.get('format')})")
            recs = [json.loads(line) for line in f if line.strip()]
        # start_s: повтор «с середины записи» (вырезка), время сдвигается так, чтобы старт был 0.
        self.frames = [r for r in recs if r["t"] >= start_s and (end_s is None or r["t"] <= end_s)]
        self.offset = start_s
        self.process_fps = float(self.header.get("process_fps", 0) or 0)

    def source(self) -> ReplaySource:
        return ReplaySource(self)

    def detector(self, source: ReplaySource) -> ReplayDetector:
        return ReplayDetector(source)


class ReplaySource:
    """Вместо VideoSource: кадр — пустышка нужного размера (одна на весь прогон), время — из записи."""

    kind = "file"

    def __init__(self, replay: Replay):
        self.replay = replay
        self.idx = -1
        self._blank: dict[tuple[int, int], np.ndarray] = {}
        from qonaq.sources.video import SourceInfo

        wh = replay.frames[0]["wh"] if replay.frames else [0, 0]
        self.info = SourceInfo(kind="file", width=wh[0], height=wh[1], fps=replay.process_fps,
                               frame_count=len(replay.frames))

    def open(self) -> ReplaySource:
        return self

    def close(self) -> None:
        pass

    def read(self) -> tuple[np.ndarray, float] | None:
        self.idx += 1
        if self.idx >= len(self.replay.frames):
            return None
        r = self.replay.frames[self.idx]
        w, h = r["wh"]
        blank = self._blank.get((w, h))
        if blank is None:
            blank = self._blank[(w, h)] = np.zeros((h, w, 3), np.uint8)
        return blank, float(r["t"]) - self.replay.offset

    def poll(self):
        return self.read()

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        while (item := self.read()) is not None:
            yield item


class ReplayDetector:
    """Детекции текущего кадра ReplaySource (с гистограммой торса в Detection.hist)."""

    name = "replay"
    device = "replay"

    def __init__(self, source: ReplaySource):
        self.source = source

    def detect(self, frames: Sequence[np.ndarray], rois=None) -> list[list[Detection]]:
        if len(frames) != 1:
            raise ValueError("повтор — одна камера, по кадру за вызов")
        r = self.source.replay.frames[self.source.idx]
        return [[Detection(p[0], p[1], p[2], p[3], p[4], PERSON, hist=_unpack_hist(p[5])) for p in r["p"]]]


def run(path: str | Path, site, zones_path: str | Path | None = None, start_s: float = 0.0,
        end_s: float | None = None, store=None) -> list:
    """Прогон пайплайна по повтору → список событий (logic.base.Event). Сигналы по кадру выключаются."""
    from qonaq.pipeline import Pipeline

    site = site.model_copy(deep=True)
    th = site.thresholds
    th.use_table_appearance = th.use_table_classifier = th.use_dish_detection = False
    site.roles.classifier = None
    if zones_path is not None:
        site.cameras[0].zones = str(zones_path)
    rp = Replay(path, start_s, end_s)
    site.fps_auto = False
    if rp.process_fps and abs(rp.process_fps - site.process_fps) > 1e-6:
        log.warning("повтор записан на %.1f кадр/с, а в конфиге %.1f — беру из повтора", rp.process_fps,
                    site.process_fps)
        site.process_fps = rp.process_fps
    src = rp.source()
    cam_id = site.cameras[0].id
    pipe = Pipeline(site, detector=rp.detector(src), store=store, sources={cam_id: src})
    events = []
    pipe.run(on_result=lambda r: events.extend(r.events))
    return events
