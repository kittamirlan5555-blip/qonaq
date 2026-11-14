"""Синтетическая «неделя» зала для отчёта пилота (S5): план посадок → наблюдения → настоящая логика столов → журнал.

Видео здесь нет (неделя видео на 2 CPU не рендерится и не детектится). Генерируется то же, что слой ролей отдаёт
логике: кто в зоне стола (гости/персонал) каждые `tick_s` секунд. Дальше работает обычный RestaurantEngine —
события, сессии и алерты те же, что в живом режиме. Время — epoch, местное время заведения (UTC+5).

Параметры (YAML, `configs/synth_week/*.yaml`): поток гостей по часам, скорость подхода официанта, время уборки,
терпение гостей, в пик всё медленнее. «Как есть» и «после» отличаются только параметрами персонала.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

from qonaq.config import SiteConfig
from qonaq.events.store import EventStore
from qonaq.logic.base import ZoneObservation
from qonaq.logic.engine import RestaurantEngine
from qonaq.zones.models import Zone, ZoneSet


class WeekParams(BaseModel):
    name: str = "week"
    description: str = ""
    seed: int = 1
    start: str = "2026-11-02"  # понедельник
    days: int = 7
    utc_offset_h: float = 5.0  # Астана
    tables: int = 10
    open_h: tuple[int, int] = (11, 23)
    # ожидаемое число посадок в час по всему залу (будни); пятница/суббота — × weekend_factor
    arrivals: dict[int, float] = Field(default_factory=lambda: {
        11: 2, 12: 6, 13: 7, 14: 4, 15: 2, 16: 2, 17: 3, 18: 7, 19: 9, 20: 8, 21: 5, 22: 2})
    weekend_factor: float = 1.35
    guests_per_table: tuple[int, int] = (1, 4)
    visit_median_s: float = 55 * 60
    visit_sigma: float = 0.3
    # первый подход официанта: логнормальное, в пик (загрузка ≥ peak_load) медленнее в peak_mult раз
    response_median_s: float = 120.0
    response_sigma: float = 0.55
    peak_load: float = 0.7
    response_peak_mult: float = 1.8
    patience_median_s: float = 600.0  # дольше ждать не станут — уйдут, не дождавшись
    # уборка после ухода
    cleanup_median_s: float = 150.0
    cleanup_sigma: float = 0.6
    cleanup_peak_mult: float = 2.0
    staff_visit_s: tuple[float, float] = (20.0, 60.0)
    extra_visits: tuple[int, int] = (3, 6)
    queue_patience_s: float = 600.0  # гости у входа ждут свободный стол не дольше
    tick_s: float = 3.0

    @classmethod
    def load(cls, path: str | Path) -> WeekParams:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class TablePlan:
    guests: list[tuple[float, float, int]] = field(default_factory=list)  # (с, по, гостей)
    staff: list[tuple[float, float]] = field(default_factory=list)
    busy_until: float = -1.0  # стол свободен и убран с этого момента


def _lognorm(rng: np.random.Generator, median: float, sigma: float) -> float:
    return float(median * math.exp(sigma * rng.standard_normal()))


def plan_day(p: WeekParams, day0: float, weekday: int, rng: np.random.Generator):
    """План одного дня: посадки по столам, визиты персонала, очередь у входа."""
    tables = [TablePlan() for _ in range(p.tables)]
    queue: list[tuple[float, float]] = []  # интервалы ожидания у входа (по одной группе)
    factor = p.weekend_factor if weekday in (4, 5) else 1.0
    arrivals: list[float] = []
    for h in range(p.open_h[0], p.open_h[1]):
        lam = p.arrivals.get(h, 0.0) * factor
        for _ in range(rng.poisson(lam)):
            arrivals.append(day0 + h * 3600 + rng.uniform(0, 3600))
    arrivals.sort()
    close = day0 + p.open_h[1] * 3600 + 1800

    def load_at(t: float) -> float:
        occ = sum(1 for tb in tables for (a, b, _) in tb.guests if a <= t < b)
        return occ / p.tables

    for t_arr in arrivals:
        free = [i for i, tb in enumerate(tables) if tb.busy_until <= t_arr]
        t_seat = t_arr
        if not free:  # ждут у входа первый освободившийся стол
            nxt = min(tables, key=lambda tb: tb.busy_until)
            if nxt.busy_until - t_arr > p.queue_patience_s:
                queue.append((t_arr, t_arr + p.queue_patience_s))
                continue
            t_seat = nxt.busy_until + rng.uniform(5, 40)
            queue.append((t_arr, t_seat))
            free = [tables.index(nxt)]
        i = int(rng.choice(free))
        tb = tables[i]
        peak = load_at(t_seat) >= p.peak_load
        resp = _lognorm(rng, p.response_median_s, p.response_sigma) * (p.response_peak_mult if peak else 1.0)
        patience = _lognorm(rng, p.patience_median_s, 0.3)
        n = int(rng.integers(p.guests_per_table[0], p.guests_per_table[1] + 1))
        if resp > patience:  # ушли, не дождавшись
            leave = t_seat + patience
            tb.guests.append((t_seat, leave, n))
            cleanup = leave + _lognorm(rng, p.cleanup_median_s * 0.5, p.cleanup_sigma)
        else:
            visit = max(15 * 60, _lognorm(rng, p.visit_median_s, p.visit_sigma))
            leave = min(t_seat + visit, close)
            tb.guests.append((t_seat, leave, n))
            v0 = t_seat + resp
            tb.staff.append((v0, v0 + rng.uniform(*p.staff_visit_s)))
            for _ in range(int(rng.integers(p.extra_visits[0], p.extra_visits[1] + 1))):
                tv = rng.uniform(v0 + 300, max(v0 + 301, leave - 120))
                tb.staff.append((tv, tv + rng.uniform(*p.staff_visit_s)))
            peak_l = load_at(leave) >= p.peak_load
            cleanup = leave + _lognorm(rng, p.cleanup_median_s, p.cleanup_sigma) * (
                p.cleanup_peak_mult if peak_l else 1.0)
        dur = rng.uniform(30, 70)
        tb.staff.append((cleanup, cleanup + dur))
        tb.busy_until = cleanup + dur + 5
    return tables, queue


def simulate(p: WeekParams, site: SiteConfig | None = None, out: str | Path | None = None) -> EventStore:
    """Неделя → EventStore (в памяти или в файл out/events.sqlite)."""
    rng = np.random.default_rng(p.seed)
    site = site or SiteConfig()
    zs = ZoneSet(camera_id="synth", frame_size=(1000, 1000),
                 zones=[Zone(id=f"T{i + 1:02d}", polygon=[(i * 90, 0), (i * 90 + 80, 0), (i * 90 + 80, 80),
                                                          (i * 90, 80)]) for i in range(p.tables)]
                 + [Zone(id="ENTRANCE", type="entrance", polygon=[(0, 900), (200, 900), (200, 999), (0, 999)])])
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        db = Path(out) / "events.sqlite"
        db.unlink(missing_ok=True)
        store = EventStore(db)
    else:
        store = EventStore(":memory:")
    tz = timezone(timedelta(hours=p.utc_offset_h))
    start = datetime.fromisoformat(p.start).replace(tzinfo=tz)
    t_first = None
    t_last = 0.0
    for d in range(p.days):
        day_dt = start + timedelta(days=d)
        day0 = day_dt.timestamp()
        tables, queue = plan_day(p, day0, day_dt.weekday(), rng)
        engine = RestaurantEngine(zs, site)  # новый день — новые машины (ночью зал закрыт)
        t0 = day0 + p.open_h[0] * 3600 - 600
        t1 = day0 + p.open_h[1] * 3600 + 3600
        t_first = t0 if t_first is None else t_first
        ts = t0
        tid = 0
        ids: dict[tuple[int, int], tuple[int, ...]] = {}
        while ts < t1:
            obs = []
            for i, tb in enumerate(tables):
                g: tuple[int, ...] = ()
                for k, (a, b, n) in enumerate(tb.guests):
                    if a <= ts < b:
                        if (i, k) not in ids:
                            ids[(i, k)] = tuple(range(tid, tid + n))
                            tid += n
                        g = ids[(i, k)]
                        break
                s = tuple(900_000 + i for (a, b) in tb.staff if a <= ts < b)[:1]
                obs.append(ZoneObservation(ts=ts, zone_id=f"T{i + 1:02d}", guest_tracks=g, staff_tracks=s,
                                           guest_confidence=0.85 if g else None))
            nq = sum(1 for a, b in queue if a <= ts < b)
            obs.append(ZoneObservation(ts=ts, zone_id="ENTRANCE", guest_tracks=tuple(range(nq))))
            evs = engine.update(obs)
            for e in evs:
                e.camera_id = "synth"
            store.add(evs)
            ts += p.tick_s
        t_last = t1
    store.set_meta(time_base="epoch", site_id=site.site_id, profile="restaurant", tables=p.tables,
                   t_start=t_first, t_end=t_last, synthetic=True, synthetic_name=p.name, utc_offset_h=p.utc_offset_h)
    return store
