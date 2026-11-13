"""Честные разбиения (W1): роли — leave-one-venue-out; столы — по камерам. Родные split-ы Roboflow не используются."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def leave_one_venue_out(venues: np.ndarray, test_venues=("A", "B", "C", "D")
                        ) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """(отложенный зал, индексы train, индексы test). Обучение — на всех остальных залах."""
    venues = np.asarray(venues)
    for v in test_venues:
        te = np.where(venues == v)[0]
        if len(te) == 0:
            continue
        tr = np.where(venues != v)[0]
        yield v, tr, te


def group_kfold(groups: np.ndarray, k: int = 5, seed: int = 0) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """K фолдов по группам (камера / блок времени): одна группа целиком в одном фолде.
    Группы раскладываются жадно по размеру, чтобы фолды были примерно равны."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    k = min(k, len(uniq))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(uniq)), -counts))
    load = np.zeros(k)
    fold_of: dict = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += counts[i]
    fold = np.array([fold_of[g] for g in groups])
    for f in range(k):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        yield f, tr, te
