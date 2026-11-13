"""Слияние независимых сигналов «стол грязный / чистый» (W3). Чистая функция, без видео.

Сигналы: `ref` — отличие от эталона пустого стола; `clf` — классификатор кропа стола; `dishes` — детектор
видит посуду в зоне без людей. Каждый: True (грязный) / False (чистый) / None (нет данных).
"""
from __future__ import annotations

from collections.abc import Iterable


def fuse(signals: Iterable[tuple[str, bool | None]], clean_min_agree: int = 2, dirty_min_agree: int = 2,
         enabled: Iterable[str] | None = None) -> bool | None:
    """True — грязный, False — чистый, None — нет согласия/данных.

    Порог согласия не больше числа включённых сигналов: с одним включённым сигналом решает он один.
    «Чистый» требует, чтобы голосов «чистый» было больше, чем «грязный» (и наоборот) — ничья = None.
    """
    sig = [(n, v) for n, v in signals if enabled is None or n in set(enabled)]
    n_on = len(sig)
    if n_on == 0:
        return None
    dirty = sum(1 for _, v in sig if v is True)
    clean = sum(1 for _, v in sig if v is False)
    if dirty >= min(dirty_min_agree, n_on) and dirty > clean:
        return True
    if clean >= min(clean_min_agree, n_on) and clean > dirty:
        return False
    return None
