"""Периоды для отчётов (S4 сводка смены, S5 пилот «как есть / стало»): срез журнала по времени и склейка прогонов.

Метрики считаются из событий (не из видео), поэтому период = новый EventStore в памяти, в который заново
проигрываются события нужного интервала; сессии столов пересобираются из них же. Сессии, начавшиеся до начала
периода, в него не попадают (консервативно: их «ушли/убрали» без «сели» игнорируются).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qonaq.events.store import EventStore
from qonaq.logic.base import Event


def _replay(dst: EventStore, events: Iterable[dict[str, Any]], offset: float = 0.0) -> int:
    n = 0
    batch = []
    for e in events:
        batch.append(Event(e["ts"] + offset, e["zone_id"], e["type"], e["payload"], e["camera_id"]))
        n += 1
        if len(batch) >= 500:
            dst.add(batch)
            batch = []
    dst.add(batch)
    return n


def slice_store(src: EventStore, t0: float, t1: float) -> EventStore:
    """События с t0 <= ts < t1 -> новый журнал в памяти; t_start/t_end = границы периода."""
    meta = src.meta()
    dst = EventStore(":memory:")
    _replay(dst, (e for e in src.events() if t0 <= e["ts"] < t1))
    dst.set_meta(**{k: v for k, v in meta.items() if k in ("time_base", "site_id", "profile", "tables")},
                 t_start=t0, t_end=t1, scene_s=t1 - t0)
    return dst


def run_span(store: EventStore) -> tuple[float, float]:
    meta = store.meta()
    ev = store.events()
    t0 = float(meta.get("t_start", 0.0)) if meta.get("time_base") != "epoch" else float(
        meta.get("t_start", min((e["ts"] for e in ev), default=0.0)))
    t1 = float(meta.get("t_end", 0.0)) or t0 + float(meta.get("scene_s", 0.0))
    t1 = max([t1] + [e["ts"] for e in ev])
    return t0, t1


def merge_runs(run_dirs: Iterable[str | Path], gap_s: float = 0.0) -> EventStore:
    """Несколько прогонов -> один журнал. Прогоны во «времени видео» (у каждого своё 0) ставятся друг за другом;
    прогоны в epoch (живые) — как есть. Число столов — максимум по прогонам."""
    dst = EventStore(":memory:")
    offset = 0.0
    tables = 0
    time_base = None
    t_first: float | None = None
    t_last = 0.0
    runs = 0
    extra: dict = {}
    for d in run_dirs:
        p = Path(d)
        db = p if p.suffix == ".sqlite" else p / "events.sqlite"
        src = EventStore(db)
        try:
            meta = src.meta()
            tb = meta.get("time_base", "video")
            time_base = time_base or tb
            t0, t1 = run_span(src)
            off = 0.0 if tb == "epoch" else offset - t0
            _replay(dst, src.events(), off)
            tables = max(tables, int(meta.get("tables") or 0))
            runs += 1
            extra.update({k: meta[k] for k in ("synthetic", "synthetic_name", "utc_offset_h", "site_id", "profile")
                          if k in meta})
            t_first = t0 + off if t_first is None else min(t_first, t0 + off)
            t_last = max(t_last, t1 + off)
            offset = t_last + gap_s
        finally:
            src.close()
    dst.set_meta(time_base=time_base or "video", tables=tables, t_start=t_first or 0.0, t_end=t_last,
                 scene_s=t_last - (t_first or 0.0), runs=runs, **extra)
    return dst
