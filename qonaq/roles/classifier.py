"""`RoleClassifier`: вероятность «персонал» по внешнему виду кропа (W2, DECISIONS D-017).

Не принимает решение о роли сам — это делает `RoleTracker`, сливая внешний вид с поведением по треку.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from qonaq.roles.head import Head

log = logging.getLogger("qonaq.roles")


@runtime_checkable
class RoleClassifier(Protocol):
    name: str

    def staff_proba(self, frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        """p(staff) для каждого bbox, shape (N,)."""
        ...


class HeadRoleClassifier:
    """Энкодер (ONNX) + HSV торса/ног + голова из `.npz`. Состав признаков берётся из метаданных головы."""

    def __init__(self, head: Head, encoder=None):
        from qonaq.roles.features import hsv_features

        self.head = head
        self._hsv = hsv_features
        self.use_hsv = bool(head.meta.get("hsv", True))
        enc_name = head.meta.get("encoder")
        if enc_name and encoder is None:
            from qonaq.roles.encoder import OnnxEncoder

            encoder = OnnxEncoder(enc_name)
        self.encoder = encoder
        self.name = f"head:{head.meta.get('features', '?')}"

    @classmethod
    def load(cls, path: str | Path) -> HeadRoleClassifier:
        return cls(Head.load(path))

    def features(self, frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
        """(X, ok): признаки и маска «кроп получился»."""
        from qonaq.roles.encoder import crop_box

        crops, idx = [], []
        for i, b in enumerate(boxes):
            c = crop_box(frame, b)
            if c is not None:
                crops.append(c)
                idx.append(i)
        parts = []
        n = len(boxes)
        ok = np.zeros(n, bool)
        ok[idx] = True
        if self.encoder is not None:
            emb = np.zeros((n, self.encoder.dim), np.float32)
            if crops:
                emb[idx] = self.encoder.embed_crops(crops)
            parts.append(emb)
        if self.use_hsv:
            parts.append(np.stack([self._hsv(frame, b) for b in boxes]) if n else np.zeros((0, 256), np.float32))
        X = np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), np.float32)
        return X, ok

    def staff_proba(self, frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        if not len(boxes):
            return np.zeros(0, np.float32)
        X, ok = self.features(frame, boxes)
        p = self.head.predict_proba(X)[:, self.head.classes.index("staff")]
        p[~ok] = np.nan
        return p.astype(np.float32)


class FakeRoleClassifier:
    """Для тестов: p(staff) по функции от bbox."""

    name = "fake"

    def __init__(self, fn):
        self.fn = fn

    def staff_proba(self, frame, boxes):
        return np.array([self.fn(b) for b in boxes], np.float32)


def load_role_classifier(path: str | Path | None) -> RoleClassifier | None:
    if not path:
        return None
    from qonaq.config import resolve_path

    p = resolve_path(str(path))
    if not p.exists():
        log.warning("голова классификатора ролей не найдена: %s — роли только по поведению", p)
        return None
    try:
        clf = HeadRoleClassifier.load(p)
    except Exception as e:  # noqa: BLE001 — нет энкодера/onnx и т.п.: работаем без внешнего вида
        log.warning("классификатор ролей не загрузился (%s) — роли только по поведению", e)
        return None
    log.info("классификатор ролей: %s (%s)", p, clf.name)
    return clf
