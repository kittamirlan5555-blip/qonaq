"""Кэш детекций на диске — инструмент разработки, не часть продукта (DECISIONS D-012).

Детектор на CPU — самое медленное звено (≈1 кадр/с в режиме качества). При отладке логики, зон и ролей
одно и то же видео прогоняется много раз, а детекции при этом не меняются. Кэш хранит ТОЛЬКО рамки
(координаты, класс, уверенность) по хэшу содержимого кадра — никаких изображений на диск не попадает.

Включение: `detector.cache_dir` в YAML объекта, флаг `--det-cache DIR` или переменная `QONAQ_DET_CACHE`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from qonaq.detect.base import Detection, Detector

log = logging.getLogger("qonaq.detect")


def frame_key(frame: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(str(frame.shape).encode())
    h.update(np.ascontiguousarray(frame).data)
    return h.hexdigest()


class CachingDetector:
    """Обёртка над любым Detector. Настоящий детектор создаётся лениво — только при промахе кэша."""

    def __init__(self, factory: Callable[[], Detector], cache_dir: str | Path, key: str):
        self._factory = factory
        self._inner: Detector | None = None
        self.name = f"{key}+cache"
        self.device = "cache"
        self.path = Path(cache_dir) / f"{key}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, list[Detection]] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # оборванная последняя строка после аварийного выхода
                        continue
                    self._mem[rec["k"]] = [Detection(*d[:5], d[5]) for d in rec["d"]]
            log.info("кэш детекций: %s (%d кадров)", self.path, len(self._mem))

    def _detector(self) -> Detector:
        if self._inner is None:
            self._inner = self._factory()
            self.device = getattr(self._inner, "device", "?")
        return self._inner

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        keys = [frame_key(f) for f in frames]
        todo = [i for i, k in enumerate(keys) if k not in self._mem]
        self.hits += len(frames) - len(todo)
        self.misses += len(todo)
        if todo:
            fresh = self._detector().detect([frames[i] for i in todo])
            with self.path.open("a", encoding="utf-8") as f:
                for i, dets in zip(todo, fresh, strict=True):
                    self._mem[keys[i]] = list(dets)
                    rec = {"k": keys[i], "d": [[round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1),
                                                round(d.conf, 4), d.label] for d in dets]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return [list(self._mem[k]) for k in keys]
