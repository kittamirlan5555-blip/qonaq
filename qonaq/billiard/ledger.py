"""Учёт бильярдных столов по видео (этап 3): рабочий день, часы игры, выручка по тарифам, загрузка по часам.

    python -m qonaq billiard ledger --events runs/live_billiard/events.sqlite --site configs/sites/billiard.yaml \
        [--date 2026-11-14 ...] [--video-start "2026-11-14 19:00"] [--out dir] [--send]

Зачем отдельно от сверки (`qonaq reconcile`). Сверке нужна выгрузка оплат из программы клуба; её может не быть
(тетрадь, касса без выгрузки) — а «сколько часов играли на каждом столе и сколько это денег по прайсу» владелец хочет
с первого дня. Учёт считается только по видео:

- **рабочий день** — с `billiard.day_start_h` (06:00) до 06:00 назавтра: ночная игра 23:30–01:10 — вчерашний день;
  сессия относится к дню своего начала целиком;
- **деньги** — поминутно по тарифу, действующему в эту минуту: `billiard.tariff_bands` (вечер/ночь/выходные, по
  столам), иначе тариф стола; как клуб берёт деньги — `bill_rounding_min` (округление вверх) и `bill_min_minutes`;
- **загрузка** — минуты игры / минуты, когда клуб открыт (`business.open_hours`) И система видела стол (heartbeat);
  минуты, когда камера не отвечала, — «не видела», в простой не идут;
- **игра вне часов работы** — отдельной строкой (после закрытия, до открытия): владельцу это важно знать.

Это оценка по видео, а не касса: скидки, абонементы, бесплатные минуты система не знает. Сверка с кассой — reconcile.
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from html import escape
from pathlib import Path
from typing import Any

from qonaq.billiard.evaluate import merge, sessions_from_events
from qonaq.strings import fmt_money, human_dur, plural, table_label

Interval = tuple[float, float]
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]


@dataclass
class Band:
    start: int  # минуты от полуночи
    end: int
    per_hour: float
    weekdays: tuple[int, ...] = ()
    tables: tuple[str, ...] = ()


def _min(hm: str) -> int:
    h, m = hm.split(":")
    return int(h) * 60 + int(m)


@dataclass
class LedgerSettings:
    tariff_per_hour: float = 2500.0
    tariffs: dict[str, float] = field(default_factory=dict)
    bands: list[Band] = field(default_factory=list)
    day_start_h: int = 6
    open_hours: str | None = "12:00-04:00"
    rounding_min: float = 0.0
    min_minutes: float = 0.0
    currency: str = "₸"
    titles: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_site(cls, site) -> LedgerSettings:
        b = site.billiard
        return cls(tariff_per_hour=b.tariff_per_hour, tariffs=dict(b.tariffs),
                   bands=[Band(_min(x.start), _min(x.end), x.per_hour, tuple(x.weekdays), tuple(x.tables))
                          for x in b.tariff_bands],
                   day_start_h=b.day_start_h, open_hours=site.business.open_hours,
                   rounding_min=b.bill_rounding_min, min_minutes=b.bill_min_minutes, currency=site.business.currency)

    def title(self, zone: str) -> str:
        return self.titles.get(zone) or table_label(zone)


# ------------------------------------------------------------------ время
def _at(day: date, minutes: int, tz: tzinfo) -> float:
    d = day + timedelta(days=minutes // 1440)
    m = minutes % 1440
    return datetime(d.year, d.month, d.day, m // 60, m % 60, tzinfo=tz).timestamp()


def workday_of(ts: float, tz: tzinfo, day_start_h: int = 6) -> date:
    """Рабочий день момента: до day_start_h — ещё вчерашний."""
    return (datetime.fromtimestamp(ts, tz) - timedelta(hours=day_start_h)).date()


def workday_window(day: date, tz: tzinfo, day_start_h: int = 6) -> Interval:
    return _at(day, day_start_h * 60, tz), _at(day + timedelta(days=1), day_start_h * 60, tz)


def open_window(day: date, open_hours: str | None, tz: tzinfo, day_start_h: int = 6) -> Interval:
    """Часы работы рабочего дня: «12:00-04:00» → 12:00 этого дня … 04:00 следующего. Нет — весь рабочий день."""
    if not open_hours:
        return workday_window(day, tz, day_start_h)
    a, b = (_min(x.strip().replace("24:00", "23:59")) for x in open_hours.split("-"))
    if a < day_start_h * 60:  # «02:00-…» открытие ночью — уже следующий календарный день рабочего дня
        a += 1440
    if b <= a % 1440 or (a >= 1440 and b < a - 1440):
        b += 1440
    if a >= 1440 and b < a:
        b += 1440
    return _at(day, a, tz), _at(day, b, tz)


# ------------------------------------------------------------------ тарифы
def rate_at(zone: str, local: datetime, cfg: LedgerSettings) -> float:
    t = local.hour * 60 + local.minute
    for b in cfg.bands:
        if b.tables and zone not in b.tables:
            continue
        if b.end > b.start:
            hit, band_day = b.start <= t < b.end, local.date()
        else:  # через полночь: 18:00–02:00 — ночь до 02:00 относится к полосе вчерашнего дня
            hit = t >= b.start or t < b.end
            band_day = local.date() if t >= b.start else local.date() - timedelta(days=1)
        if hit and (not b.weekdays or band_day.isoweekday() in b.weekdays):
            return float(b.per_hour)
    return float(cfg.tariffs.get(zone, cfg.tariff_per_hour))


def price(zone: str, a: float, b: float, cfg: LedgerSettings, tz: tzinfo) -> tuple[float, dict[float, float]]:
    """Сумма за [a, b] поминутно по действующему тарифу и минуты по каждому тарифу."""
    amount, by_rate = 0.0, {}
    t = a
    while t < b:
        nxt = min(b, (math.floor(t / 60.0) + 1) * 60.0)
        r = rate_at(zone, datetime.fromtimestamp(t, tz), cfg)
        amount += r * (nxt - t) / 3600.0
        by_rate[r] = by_rate.get(r, 0.0) + (nxt - t) / 60.0
        t = nxt
    return amount, by_rate


def billed_minutes(minutes: float, cfg: LedgerSettings) -> float:
    m = minutes
    if cfg.rounding_min > 0:
        m = math.ceil(m / cfg.rounding_min - 1e-9) * cfg.rounding_min
    return max(m, cfg.min_minutes) if minutes > 0 else 0.0


# ------------------------------------------------------------------ расчёт
def _clip(iv: list[Interval], a: float, b: float) -> list[Interval]:
    return [(max(x, a), min(y, b)) for x, y in iv if min(y, b) > max(x, a)]


def _total(iv: list[Interval]) -> float:
    return sum(b - a for a, b in merge(iv))


def _inter(x: list[Interval], y: list[Interval]) -> list[Interval]:
    out = []
    for a, b in merge(x):
        for c, d in merge(y):
            if min(b, d) > max(a, c):
                out.append((max(a, c), min(b, d)))
    return out


def build_ledger(sessions: dict[str, list[Interval]], cfg: LedgerSettings, tz: tzinfo,
                 coverage: list[Interval] | None = None, tables: set[str] | None = None,
                 open_zones: dict[str, float] | None = None, days: list[date] | None = None) -> list[dict[str, Any]]:
    """Учёт по рабочим дням. sessions — epoch-интервалы игры по столам; coverage — когда система видела (None —
    всегда); tables — все столы объекта (стол без игр тоже в отчёте: простой целый день); open_zones — стол: момент,
    до которого посчитана ещё идущая игра."""
    open_zones = open_zones or {}
    by_day: dict[date, list[tuple[str, float, float]]] = {}
    for z, iv in sessions.items():
        for a, b in merge(iv):
            by_day.setdefault(workday_of(a, tz, cfg.day_start_h), []).append((z, a, b))
    all_days = sorted(set(by_day) | set(days or []))
    if days:
        all_days = [d for d in all_days if d in set(days)]
    zones = sorted(set(tables or set()) | set(sessions))
    return [_day(d, by_day.get(d, []), zones, cfg, tz, coverage, open_zones) for d in all_days]


def _day(day: date, rows: list[tuple[str, float, float]], zones: list[str], cfg: LedgerSettings, tz: tzinfo,
         coverage: list[Interval] | None, open_zones: dict[str, float]) -> dict[str, Any]:
    w0, w1 = workday_window(day, tz, cfg.day_start_h)
    o0, o1 = open_window(day, cfg.open_hours, tz, cfg.day_start_h)
    seen = merge(_clip(coverage, w0, w1)) if coverage is not None else [(w0, w1)]
    open_seen = _inter([(o0, o1)], seen)
    open_seen_s = _total(open_seen)
    blind_open_s = (o1 - o0) - open_seen_s
    sess, tables = [], []
    span_a = min([o0] + [a for _, a, _ in rows])
    span_b = max([o1] + [b for _, _, b in rows])
    h0 = datetime.fromtimestamp(span_a, tz).replace(minute=0, second=0, microsecond=0).timestamp()
    n_hours = max(1, math.ceil((span_b - h0) / 3600.0))
    hours = {z: [0.0] * n_hours for z in zones}
    for z, a, b in sorted(rows, key=lambda r: r[1]):
        minutes = (b - a) / 60.0
        amount, by_rate = price(z, a, b, cfg, tz)
        bm = billed_minutes(minutes, cfg)
        amount_b = amount * bm / minutes if minutes > 0 else 0.0
        outside = (_total(_clip([(a, b)], -1e18, o0)) + _total(_clip([(a, b)], o1, 1e18))) / 60.0
        still = z in open_zones and abs(open_zones[z] - b) < 1.0
        sess.append({"zone": z, "title": cfg.title(z), "start": a, "end": b, "minutes": round(minutes, 1),
                     "billed_minutes": round(bm, 1), "amount": round(amount_b), "by_rate": {
                         str(int(k)): round(v, 1) for k, v in sorted(by_rate.items())},
                     "outside_open_min": round(outside, 1), "still_playing": still})
        for i in range(n_hours):
            s0 = h0 + i * 3600
            hours[z][i] += max(0.0, min(b, s0 + 3600) - max(a, s0)) / 60.0
    for z in zones:
        mine = [s for s in sess if s["zone"] == z]
        iv = [(s["start"], s["end"]) for s in mine]
        play_open_s = _total(_inter(iv, open_seen))
        tables.append({
            "zone": z, "title": cfg.title(z), "games": len(mine),
            "play_min": round(sum(s["minutes"] for s in mine), 1),
            "amount": sum(s["amount"] for s in mine),
            "util": round(play_open_s / open_seen_s, 3) if open_seen_s > 0 else None,
            "idle_open_min": round(max(0.0, open_seen_s - play_open_s) / 60.0, 1),
            "longest_min": max((s["minutes"] for s in mine), default=0.0),
            "first": min((s["start"] for s in mine), default=None),
            "last": max((s["end"] for s in mine), default=None),
            "outside_open_min": round(sum(s["outside_open_min"] for s in mine), 1),
        })
    tot_h = [sum(hours[z][i] for z in zones) for i in range(n_hours)]
    peak = max(range(n_hours), key=lambda i: tot_h[i]) if any(tot_h) else None
    notes = []
    for s in sess:
        if s["still_playing"]:
            notes.append(f"{s['title']}: игра ещё идёт — посчитана до {_hm(s['end'], tz)}")
    if blind_open_s >= 60:
        notes.append(f"В часы работы система не видела зал {human_dur(blind_open_s)} — эти минуты не в учёте")
    out_min = sum(t["outside_open_min"] for t in tables)
    if out_min >= 1:
        notes.append(f"Игра вне часов работы ({cfg.open_hours}): {human_dur(out_min * 60)}")
    n_open = len(zones) * open_seen_s
    play_open_total = sum(_total(_inter([(s["start"], s["end"]) for s in sess if s["zone"] == z], open_seen))
                          for z in zones)
    return {
        "day": day.isoformat(), "window": [w0, w1], "open": [o0, o1], "open_hours": cfg.open_hours,
        "currency": cfg.currency, "tables": tables, "sessions": sess,
        "hours": {"start": h0, "labels": [_hm(h0 + i * 3600, tz) for i in range(n_hours)],
                  "by_table": {z: [round(v, 1) for v in hours[z]] for z in zones},
                  "total": [round(v, 1) for v in tot_h]},
        "peak": None if peak is None else {"label": f"{_hm(h0 + peak * 3600, tz)}–{_hm(h0 + (peak + 1) * 3600, tz)}",
                                           "minutes": round(tot_h[peak], 1)},
        "totals": {"games": len(sess), "play_min": round(sum(s["minutes"] for s in sess), 1),
                   "amount": sum(s["amount"] for s in sess), "tables": len(zones),
                   "util": round(play_open_total / n_open, 3) if n_open > 0 else None,
                   "blind_open_min": round(blind_open_s / 60.0, 1)},
        "notes": notes,
    }


# ------------------------------------------------------------------ вывод
def _hm(ts: float, tz: tzinfo) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%H:%M")


def _dur(minutes: float) -> str:
    return human_dur(round(minutes) * 60) if minutes >= 1 else "—"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{round(x * 100)}%"


def day_label(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d.day} {MONTHS[d.month - 1]}"


def telegram_text(d: dict[str, Any], site_name: str = "", max_tables: int = 8) -> str:
    t, cur = d["totals"], d["currency"]
    head = f"Бильярд · {day_label(d['day'])}" + (f" · {site_name}" if site_name else "")
    lines = [head]
    if not t["games"]:
        lines.append("Игр по видео не было")
    else:
        n_played = sum(1 for x in d["tables"] if x["games"])
        lines.append(f"Игра: {_dur(t['play_min'])} на {plural(n_played, 'столе', 'столах', 'столах')}"
                     f", {plural(t['games'], 'сессия', 'сессии', 'сессий')}")
        lines.append(f"По прайсу ≈ {fmt_money(t['amount'], cur)}")
        if t["util"] is not None:
            lines.append(f"Загрузка в часы работы: {_pct(t['util'])}" +
                         (f", пик {d['peak']['label']}" if d["peak"] else ""))
        for x in sorted(d["tables"], key=lambda x: -x["play_min"])[:max_tables]:
            lines.append(f"{x['title']} — {_dur(x['play_min'])} · {fmt_money(x['amount'], cur)}" if x["games"]
                         else f"{x['title']} — не играли")
    lines += d["notes"]
    lines.append("Это оценка по видео: скидки и абонементы система не знает.")
    return "\n".join(lines)


def sessions_csv(days: list[dict[str, Any]], tz: tzinfo) -> str:
    """Для Excel: `;`, UTF-8 с BOM, деньги целыми."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    w.writerow(["Рабочий день", "Стол", "Начало", "Конец", "Минут", "К оплате, мин", "Сумма", "Тарифы (₸/ч: мин)",
                "Вне часов работы, мин", "Идёт"])
    for d in days:
        for s in d["sessions"]:
            w.writerow([d["day"], s["title"], datetime.fromtimestamp(s["start"], tz).strftime("%Y-%m-%d %H:%M"),
                        datetime.fromtimestamp(s["end"], tz).strftime("%Y-%m-%d %H:%M"),
                        _num(s["minutes"]), _num(s["billed_minutes"]), s["amount"],
                        ", ".join(f"{k}: {_num(v)}" for k, v in s["by_rate"].items()), _num(s["outside_open_min"]),
                        "да" if s["still_playing"] else ""])
    return "﻿" + buf.getvalue()


def tables_csv(days: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    w.writerow(["Рабочий день", "Стол", "Сессий", "Минут игры", "Сумма", "Загрузка, %", "Простой в часы работы, мин",
                "Самая длинная, мин"])
    for d in days:
        for x in d["tables"]:
            w.writerow([d["day"], x["title"], x["games"], _num(x["play_min"]), x["amount"],
                        "" if x["util"] is None else round(x["util"] * 100), _num(x["idle_open_min"]),
                        _num(x["longest_min"])])
    return "﻿" + buf.getvalue()


def _num(x: float) -> str:
    return f"{x:.1f}".replace(".", ",").replace(",0", "") if x != int(x) else str(int(x))


def html_report(days: list[dict[str, Any]], tz: tzinfo, title: str = "") -> str:
    css = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--mut:#667085;--line:#e4e7ec;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1217;--card:#171b22;--ink:#e8eaee;--mut:#98a2b3;--line:#2a303a}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:19px;margin:32px 0 10px}.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);
border-radius:14px;padding:16px;margin:12px 0;overflow-x:auto}.kpi{display:grid;grid-template-columns:repeat(auto-fit,
minmax(150px,1fr));gap:12px}.kpi div{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:14px}
.kpi b{display:block;font-size:24px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mut);
font-weight:600;font-size:13px}td.n{text-align:right}.heat td{padding:0;border:0}.heat .c{width:34px;height:26px;
border-radius:4px;margin:1px;display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff}
.heat th{font-size:11px;padding:2px}.note{margin:4px 0}"""
    out = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Учёт столов</title><style>{css}</style><main>"
           f"<h1>Учёт бильярдных столов{(' · ' + escape(title)) if title else ''}</h1>"
           "<div class=mut>По видео: часы игры и сумма по прайсу. Скидки, абонементы и бесплатные минуты система не "
           "знает — для сверки с кассой есть отдельный отчёт.</div>"]
    for d in days:
        t, cur = d["totals"], d["currency"]
        out.append(f"<h2>{day_label(d['day'])} <span class=mut>· часы работы {escape(d['open_hours'] or '—')}</span>"
                   "</h2><div class=kpi>"
                   f"<div><span class=mut>Игра</span><b>{_dur(t['play_min'])}</b></div>"
                   f"<div><span class=mut>Сессий</span><b>{t['games']}</b></div>"
                   f"<div><span class=mut>По прайсу</span><b>{escape(fmt_money(t['amount'], cur))}</b></div>"
                   f"<div><span class=mut>Загрузка</span><b>{_pct(t['util'])}</b></div>"
                   f"<div><span class=mut>Пик</span><b>{escape(d['peak']['label']) if d['peak'] else '—'}</b></div>"
                   "</div>")
        for n in d["notes"]:
            out.append(f"<div class='note mut'>• {escape(n)}</div>")
        out.append("<div class=card><table><tr><th>Стол<th>Сессий<th>Игра<th>Сумма<th>Загрузка<th>Простой<th>"
                   "Первая — последняя</tr>")
        for x in d["tables"]:
            span = f"{_hm(x['first'], tz)} — {_hm(x['last'], tz)}" if x["first"] else "—"
            out.append(f"<tr><td>{escape(x['title'])}<td class=n>{x['games']}<td>{_dur(x['play_min'])}"
                       f"<td class=n>{escape(fmt_money(x['amount'], cur))}<td class=n>{_pct(x['util'])}"
                       f"<td>{_dur(x['idle_open_min'])}<td>{span}</tr>")
        out.append("</table></div>")
        hs = d["hours"]
        out.append("<div class=card><div class=mut style='margin-bottom:6px'>Минуты игры по часам</div>"
                   "<table class=heat><tr><th></th>" + "".join(f"<th>{lb}</th>" for lb in hs["labels"]) + "</tr>")
        for z, row in hs["by_table"].items():
            cells = "".join(
                f"<td><div class=c style='background:rgba(37,99,235,{0.08 + 0.92 * min(v, 60) / 60:.2f})'>"
                f"{round(v) if v >= 1 else ''}</div></td>" for v in row)
            title_z = next(x["title"] for x in d["tables"] if x["zone"] == z)
            out.append(f"<tr><th style='text-align:right;padding-right:6px'>{escape(title_z)}</th>{cells}</tr>")
        out.append("</table></div>")
        if d["sessions"]:
            out.append("<div class=card><table><tr><th>Стол<th>Начало<th>Конец<th>Минут<th>К оплате<th>Сумма"
                       "<th>Тарифы</tr>")
            for s in d["sessions"]:
                tag = " (идёт)" if s["still_playing"] else ""
                out.append(f"<tr><td>{escape(s['title'])}<td>{_hm(s['start'], tz)}<td>{_hm(s['end'], tz)}{tag}"
                           f"<td class=n>{_num(s['minutes'])}<td class=n>{_num(s['billed_minutes'])}"
                           f"<td class=n>{escape(fmt_money(s['amount'], cur))}<td class=mut>"
                           + escape(", ".join(f"{k} ₸/ч × {_num(v)} мин" for k, v in s["by_rate"].items())) + "</tr>")
            out.append("</table></div>")
    out.append("</main>")
    return "\n".join(out)


def write_outputs(days: list[dict[str, Any]], out: Path, tz: tzinfo, title: str = "") -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "ledger.json", "html": out / "ledger.html", "sessions": out / "ledger_sessions.csv",
             "tables": out / "ledger_tables.csv", "telegram": out / "ledger_telegram.txt"}
    paths["json"].write_text(json.dumps(days, ensure_ascii=False, indent=1), encoding="utf-8")
    paths["html"].write_text(html_report(days, tz, title), encoding="utf-8")
    paths["sessions"].write_text(sessions_csv(days, tz), encoding="utf-8")
    paths["tables"].write_text(tables_csv(days), encoding="utf-8")
    paths["telegram"].write_text("\n\n".join(telegram_text(d, title) for d in days), encoding="utf-8")
    return paths


# ------------------------------------------------------------------ из журнала
def from_store(store, cfg: LedgerSettings, tz: tzinfo, offset: float = 0.0, tables: set[str] | None = None,
               days: list[date] | None = None) -> list[dict[str, Any]]:
    """events.sqlite (live: epoch; прогон файла: секунды видео + offset) → учёт по рабочим дням."""
    from qonaq.reconcile import coverage_from_store

    evs = store.events(types=["game_started", "game_ended"])
    cov = coverage_from_store(store, offset).get("*")
    last_seen = max([b for _, b in cov] if cov else [] + [e["ts"] + offset for e in evs[-1:]] or [0.0])
    raw = sessions_from_events(evs, end_ts=last_seen - offset)
    sessions = {z: [(a + offset, b + offset) for a, b in iv] for z, iv in raw.items()}
    last_type: dict[str, str] = {}
    for e in sorted(evs, key=lambda e: e["ts"]):
        last_type[e["zone_id"]] = e["type"]
    open_zones = {z: last_seen for z, t in last_type.items() if t == "game_started"}
    return build_ledger(sessions, cfg, tz, cov, tables, open_zones, days)
