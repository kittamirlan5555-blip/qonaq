"""Отчёт пилота «как есть / стало» для владельца заведения (S5).

    python -m qonaq report pilot --baseline runs/live_cafe --baseline-from 2026-11-02 --baseline-to 2026-11-08 \\
                                 --after runs/live_cafe --after-from 2026-11-16 --after-to 2026-11-29 --site ...

Периоды — журналы событий (папки прогонов или events.sqlite, можно несколько) и/или интервал дат. Метрики — из
событий (metrics/compute.py), без видео. Деньги — консервативно: только лишний простой неубранных столов в часы
пик (формула в отчёте), уходы «не дождавшись» — отдельной строкой.
"""
from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qonaq.config import SiteConfig
from qonaq.events.store import EventStore
from qonaq.metrics.compute import compute_metrics, percentile
from qonaq.metrics.period import merge_runs, slice_store
from qonaq.metrics.report import dur
from qonaq.strings import fmt_money, human_dur, plural, table_label

MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]


def _tz(site: SiteConfig, store: EventStore):
    off = store.meta().get("utc_offset_h")
    if off is not None:
        return timezone(timedelta(hours=float(off)))
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(site.business.timezone)
    except Exception:  # noqa: BLE001
        return timezone(timedelta(hours=5))


def _parse_day(s: str, tz, end: bool = False) -> float:
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=tz)
    if end and len(s) <= 10:  # дата без времени: включительно до конца дня
        d += timedelta(days=1)
    return d.timestamp()


def load_period(paths: list[str], t_from: str | None, t_to: str | None, site: SiteConfig) -> EventStore:
    store = merge_runs(paths)
    if t_from or t_to:
        tz = _tz(site, store)
        meta = store.meta()
        a = _parse_day(t_from, tz) if t_from else float(meta.get("t_start", 0.0))
        b = _parse_day(t_to, tz, end=True) if t_to else float(meta.get("t_end", 0.0))
        sliced = slice_store(store, a, b)
        for k in ("synthetic", "synthetic_name", "utc_offset_h"):
            if k in meta:
                sliced.set_meta(**{k: meta[k]})
        store.close()
        store = sliced
    return store


def analyze(store: EventStore, site: SiteConfig) -> dict[str, Any]:
    m = compute_metrics(store, site)
    meta = store.meta()
    epoch = meta.get("time_base") == "epoch"
    tz = _tz(site, store)
    sessions = [s for s in store.sessions("table") if not s["preexisting"]]
    resp = [s["response_time_s"] for s in sessions if s["response_time_s"] is not None]
    if epoch:
        days = len({datetime.fromtimestamp(s["seated_ts"], tz).date() for s in sessions}) or 1
        t0, t1 = m["period"]["t0"], m["period"]["t_end"]
        label = (f"{_date(t0, tz)} – {_date(t1 - 1, tz)}")
    else:
        days = max(1, int(meta.get("runs") or 1))
        label = f"{plural(days, 'запись', 'записи', 'записей')}, {dur(m['period']['duration_s'])}"
    thr = site.alerts.no_waiter_s
    by_hour: dict[int, dict[str, Any]] = {}
    if epoch:
        for h in range(24):
            r = [s["response_time_s"] for s in sessions if s["response_time_s"] is not None
                 and datetime.fromtimestamp(s["seated_ts"], tz).hour == h]
            c = [s["cleanup_time_s"] for s in sessions if s["cleanup_time_s"] is not None and s["left_ts"]
                 and datetime.fromtimestamp(s["left_ts"], tz).hour == h]
            w = sum(1 for s in sessions if s["outcome"] == "walkout" and datetime.fromtimestamp(s["seated_ts"],
                                                                                                tz).hour == h)
            if r or c or w:
                by_hour[h] = {"n": len(r), "median": percentile(r, 50), "over": sum(1 for x in r if x > thr),
                              "share_over": (sum(1 for x in r if x > thr) / len(r)) if r else None,
                              "cleanup_median": percentile(c, 50), "walkouts": w}
    roi = m["roi"]
    return {
        "m": m, "days": days, "label": label, "epoch": epoch, "resp": resp, "by_hour": by_hour,
        "median": m["response"]["median"], "p90": m["response"]["p90"],
        "share_over": m["response"]["share_over_threshold"],
        "seatings_day": m["turnover"]["seatings"] / days,
        "walkouts_day": m["walkouts"]["n"] / days,
        "dirty_peak_min_day": roi["excess_dirty_in_peak_s"] / 60 / days,
        "peak_h_day": roi["peak_s"] / 3600 / days,
        "loss_day": roi["loss_kzt"] / days,
        "walkout_loss_day": roi["walkout_loss_kzt"] / days,
        "synthetic": bool(meta.get("synthetic")),
        "low_conf": m["response"].get("low_confidence_share"),
        "events": m["quality"]["events"],
    }


def _date(ts: float, tz) -> str:
    d = datetime.fromtimestamp(ts, tz)
    return f"{d.day} {MONTHS[d.month - 1]}"


def bootstrap_median_diff(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95% интервал для разницы медиан (b − a) — чтобы не выдавать шум за улучшение."""
    if len(a) < 10 or len(b) < 10:
        return None
    rng = np.random.default_rng(seed)
    A, B = np.asarray(a, float), np.asarray(b, float)
    d = [np.median(rng.choice(B, len(B))) - np.median(rng.choice(A, len(A))) for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------ HTML
def _delta(a: float | None, b: float | None, lower_is_better: bool = True, fmt=None, pct_points: bool = False) -> str:
    if a is None or b is None:
        return '<span class="muted">—</span>'
    d = b - a
    if abs(d) < 1e-9:
        return '<span class="muted">без изменений</span>'
    good = (d < 0) == lower_is_better
    if pct_points:
        txt = f"{'+' if d > 0 else '−'}{abs(d) * 100:.0f} п.п."
    elif fmt is not None:
        txt = ("+" if d > 0 else "−") + fmt(abs(d))
    else:
        txt = f"{'+' if d > 0 else '−'}{abs(d):.1f}"
    rel = f" ({'+' if d > 0 else '−'}{abs(d) / abs(a) * 100:.0f}%)" if a and not pct_points else ""
    return f'<span class="{"good" if good else "bad"}">{"▼" if d < 0 else "▲"} {html.escape(txt + rel)}</span>'


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def _hours_svg(base: dict[int, dict], after: dict[int, dict], thr: float) -> str:
    hours = sorted(set(base) | set(after))
    hours = [h for h in hours if (base.get(h, {}).get("n") or after.get(h, {}).get("n"))]
    if not hours:
        return ""
    vals = [v.get("median") or 0 for d in (base, after) for v in d.values()]
    ymax = max(max(vals), thr) * 1.15
    W, H, L, B, T = 720, 260, 44, 34, 14
    pw, ph = W - L - 10, H - B - T
    gw = pw / len(hours)
    bw = min(18.0, (gw - 8) / 2)
    y = lambda v: T + ph - ph * v / ymax  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Подход официанта по часам" class="chart">']
    step = 60 if ymax <= 420 else 120
    for k in range(0, int(ymax) + 1, step):
        out.append(f'<line x1="{L}" x2="{W - 10}" y1="{y(k):.1f}" y2="{y(k):.1f}" class="grid"/>'
                   f'<text x="{L - 6}" y="{y(k) + 4:.1f}" class="ax" text-anchor="end">{k // 60} мин</text>')
    out.append(f'<line x1="{L}" x2="{W - 10}" y1="{y(thr):.1f}" y2="{y(thr):.1f}" class="thr"/>'
               f'<text x="{W - 12}" y="{y(thr) - 5:.1f}" class="ax" text-anchor="end">порог {human_dur(thr)}</text>')
    for i, h in enumerate(hours):
        cx = L + gw * i + gw / 2
        for j, (d, cls, name) in enumerate(((base, "s1", "как было"), (after, "s2", "стало"))):
            v = d.get(h, {}).get("median")
            if v is None:
                continue
            x = cx - bw - 1 + j * (bw + 2)
            top = y(v)
            r = min(4.0, bw / 2)
            hgt = T + ph - top
            path = (f"M{x:.1f},{T + ph:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} "
                    f"H{x + bw - r:.1f} Q{x + bw:.1f},{top:.1f} {x + bw:.1f},{top + r:.1f} V{T + ph:.1f} Z") \
                if hgt > r else f"M{x:.1f},{T + ph:.1f} h{bw:.1f} v{-hgt:.1f} h{-bw:.1f} Z"
            n = d[h]["n"]
            out.append(f'<path d="{path}" class="{cls}"><title>{h:02d}:00–{h + 1:02d}:00 · {name}: '
                       f'обычно {dur(v)}, посадок {n}</title></path>')
        out.append(f'<text x="{cx:.1f}" y="{H - B + 16}" class="ax" text-anchor="middle">{h}</text>')
    out.append(f'<text x="{L + pw / 2:.1f}" y="{H - 4}" class="ax" text-anchor="middle">час посадки</text>')
    out.append("</svg>")
    return "".join(out)


def _worst_hours(a: dict[int, dict], k: int = 3) -> list[tuple[int, dict]]:
    rows = [(h, v) for h, v in a.items() if v["n"] >= 3 and v["median"] is not None]
    return sorted(rows, key=lambda x: (-(x[1]["share_over"] or 0), -(x[1]["median"] or 0)))[:k]


CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --card:#ffffff; --s1:#2a78d6; --s2:#eb6834;
  --good:#0a7d0a; --bad:#c0392b; --soft:#f3f2ee; --warn:#fff4d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --line:#383835; --card:#222220; --s1:#3987e5; --s2:#d95926; --good:#4cc24c; --bad:#ff7b6b; --soft:#2a2a28;
  --warn:#3a3218; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 "Segoe UI",Roboto,Arial,sans-serif; }
.page { max-width:900px; margin:0 auto; padding:24px 16px 40px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:18px; margin:28px 0 10px; }
.muted { color:var(--ink2); } .good { color:var(--good); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.banner { background:var(--warn); border-radius:8px; padding:10px 14px; margin:14px 0; font-weight:600; }
.lead { font-size:17px; margin:14px 0; }
table { width:100%; border-collapse:collapse; background:var(--card); }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--ink2); font-weight:600; font-size:13px; }
td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:12px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.card .v { font-size:24px; font-weight:700; } .card .k { color:var(--ink2); font-size:13px; }
.chart { width:100%; height:auto; } .chart .grid { stroke:var(--line); stroke-width:1; }
.chart .thr { stroke:var(--ink2); stroke-width:1.5; stroke-dasharray:5 4; }
.chart .ax { fill:var(--ink2); font-size:11px; } .chart .s1 { fill:var(--s1); } .chart .s2 { fill:var(--s2); }
.chart path:hover { opacity:.8; }
.legend { display:flex; gap:16px; font-size:13px; color:var(--ink2); margin:4px 0 8px; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.small { font-size:12.5px; color:var(--ink2); } ul { padding-left:20px; }
details { margin-top:8px; } summary { cursor:pointer; color:var(--ink2); font-size:13px; }
@media print { .banner { border:1px solid #999; } body { font-size:12px; } }
"""


def pilot_html(site: SiteConfig, A: dict[str, Any], Bp: dict[str, Any] | None) -> str:
    cur = site.business.currency
    thr = site.alerts.no_waiter_s
    e = html.escape
    synthetic = A["synthetic"] or (Bp or {}).get("synthetic")
    P = []
    P.append(f"<h1>Пилот Qonaq Vision · {e(site.name)}</h1>")
    P.append(f'<div class="muted">Как было: {e(A["label"])}' + (f' · Стало: {e(Bp["label"])}' if Bp else "")
             + f' · средний чек на стол {fmt_money(site.business.avg_check_kzt, cur)}</div>')
    if synthetic:
        P.append('<div class="banner">ПРИМЕР НА СИНТЕТИЧЕСКИХ ДАННЫХ. Это не реальное заведение: посадки и работа '
                 'персонала сгенерированы, чтобы показать, как выглядит отчёт.</div>')
    if Bp:
        lead = []
        if A["median"] is not None and Bp["median"] is not None:
            better = Bp["median"] < A["median"]
            lead.append(f"Официант стал подходить {'быстрее' if better else 'медленнее'}: обычно за "
                        f"<b>{dur(Bp['median'])}</b> вместо {dur(A['median'])}.")
        if A["share_over"] is not None and Bp["share_over"] is not None:
            lead.append(f"Ждали дольше {human_dur(thr)}: <b>{_pct(Bp['share_over'])}</b> столов против "
                        f"{_pct(A['share_over'])}.")
        if A["loss_day"] or Bp["loss_day"]:
            lead.append(f"Оценка упущенной выручки в часы пик: <b>{fmt_money(Bp['loss_day'], cur)}</b> в день "
                        f"против {fmt_money(A['loss_day'], cur)}.")
        P.append(f'<p class="lead">{" ".join(lead)}</p>')
    # карточки
    src = Bp or A
    P.append('<div class="cards">')
    P.append(f'<div class="card"><div class="k">Подход официанта, обычно</div><div class="v">{dur(src["median"])}'
             f'</div>{_delta(A["median"], Bp["median"], fmt=dur) if Bp else ""}</div>')
    P.append(f'<div class="card"><div class="k">Ждали дольше {human_dur(thr)}</div><div class="v">'
             f'{_pct(src["share_over"])}</div>'
             f'{_delta(A["share_over"], Bp["share_over"], pct_points=True) if Bp else ""}</div>')
    P.append(f'<div class="card"><div class="k">Неубранные столы в пик, лишних минут в день</div><div class="v">'
             f'{src["dirty_peak_min_day"]:.0f}</div>'
             f'{_delta(A["dirty_peak_min_day"], Bp["dirty_peak_min_day"], fmt=lambda v: f"{v:.0f} мин") if Bp else ""}'
             f'</div>')
    P.append(f'<div class="card"><div class="k">Упущенная выручка (пик), в день</div><div class="v">'
             f'{fmt_money(src["loss_day"], cur)}</div>'
             f'{_delta(A["loss_day"], Bp["loss_day"], fmt=lambda v: fmt_money(v, cur)) if Bp else ""}</div>')
    P.append("</div>")
    # таблица сравнения
    rows = [
        ("Подход официанта — обычно (медиана)", dur(A["median"]), dur(Bp["median"]) if Bp else None,
         _delta(A["median"], Bp["median"], fmt=dur) if Bp else ""),
        ("Подход официанта — у худших 10% столов (90-й перцентиль)", dur(A["p90"]), dur(Bp["p90"]) if Bp else None,
         _delta(A["p90"], Bp["p90"], fmt=dur) if Bp else ""),
        (f"Доля столов, ждавших дольше {human_dur(thr)}", _pct(A["share_over"]),
         _pct(Bp["share_over"]) if Bp else None, _delta(A["share_over"], Bp["share_over"], pct_points=True)
         if Bp else ""),
        ("Ушли, не дождавшись официанта — в день", f"{A['walkouts_day']:.1f}",
         f"{Bp['walkouts_day']:.1f}" if Bp else None, _delta(A["walkouts_day"], Bp["walkouts_day"]) if Bp else ""),
        ("Лишний простой неубранных столов в часы пик — минут в день", f"{A['dirty_peak_min_day']:.0f}",
         f"{Bp['dirty_peak_min_day']:.0f}" if Bp else None,
         _delta(A["dirty_peak_min_day"], Bp["dirty_peak_min_day"], fmt=lambda v: f"{v:.0f}") if Bp else ""),
        ("Часы пик (зал загружен) — часов в день", f"{A['peak_h_day']:.1f}", f"{Bp['peak_h_day']:.1f}" if Bp else None,
         '<span class="muted">для справки</span>' if Bp else ""),
        ("Посадок за столы — в день", f"{A['seatings_day']:.0f}", f"{Bp['seatings_day']:.0f}" if Bp else None,
         _delta(A["seatings_day"], Bp["seatings_day"], lower_is_better=False) if Bp else ""),
        ("Оценка упущенной выручки в часы пик — в день", fmt_money(A["loss_day"], cur),
         fmt_money(Bp["loss_day"], cur) if Bp else None,
         _delta(A["loss_day"], Bp["loss_day"], fmt=lambda v: fmt_money(v, cur)) if Bp else ""),
        ("То же — в месяц (×30, если так каждый день)", fmt_money(A["loss_day"] * 30, cur),
         fmt_money(Bp["loss_day"] * 30, cur) if Bp else None, ""),
    ]
    P.append("<h2>Как было и как стало</h2><table><tr><th>Показатель</th><th>Как было</th>"
             + ("<th>Стало</th><th>Изменение</th>" if Bp else "") + "</tr>")
    for name, a, b, d in rows:
        P.append(f"<tr><td>{e(name)}</td><td class=num>{e(a)}</td>"
                 + (f"<td class=num>{e(b)}</td><td>{d}</td>" if Bp else "") + "</tr>")
    P.append("</table>")
    if Bp:
        ci = bootstrap_median_diff(A["resp"], Bp["resp"])
        if ci:
            sure = ci[1] < 0 or ci[0] > 0
            P.append(f'<p class="small">Насколько уверенно: разница медиан подхода — от {"+" if ci[0] > 0 else "−"}'
                     f'{dur(abs(ci[0]))} до {"+" if ci[1] > 0 else "−"}{dur(abs(ci[1]))} (95% интервал, бутстрэп по '
                     f'{len(A["resp"])} и {len(Bp["resp"])} посадкам). '
                     + ("Интервал не включает ноль — изменение не похоже на случайность."
                        if sure else "Интервал включает ноль — по этим данным изменение может быть случайным.")
                     + " Это про случайный разброс, а не про ошибки распознавания (о них — ниже).</p>")
        if A["walkout_loss_day"] or Bp["walkout_loss_day"]:
            P.append(f'<p class="small">Отдельно: ушли, не дождавшись — {fmt_money(A["walkout_loss_day"], cur)} → '
                     f'{fmt_money(Bp["walkout_loss_day"], cur)} в день по среднему чеку. В итог не включено: '
                     'уход без заказа система видит хуже всего, и это самая неуверенная цифра.</p>')
    # часы
    if A["by_hour"] or (Bp and Bp["by_hour"]):
        P.append("<h2>Подход официанта по часам</h2>")
        P.append('<div class="legend"><span><i style="background:var(--s1)"></i>как было</span>'
                 + ('<span><i style="background:var(--s2)"></i>стало</span>' if Bp else "") + "</div>")
        P.append(_hours_svg(A["by_hour"], Bp["by_hour"] if Bp else {}, thr))
        P.append('<p class="small">Высота столбика — обычное (медианное) время до первого подхода для посадок в этот '
                 'час. Наведите на столбик — точное значение.</p>')
        for title, X in (("Худшие часы — как было", A), ("Худшие часы — стало", Bp)):
            if not X:
                continue
            wh = _worst_hours(X["by_hour"])
            if wh:
                items = "".join(f"<li><b>{h:02d}:00–{h + 1:02d}:00</b>: {v['over']} из {v['n']} столов ждали "
                                f"дольше {human_dur(thr)}; обычно подход за {dur(v['median'])}</li>" for h, v in wh)
                P.append(f"<h3>{title}</h3><ul>{items}</ul>")
        hours = sorted(set(A["by_hour"]) | set((Bp or {}).get("by_hour", {})))
        trs = "".join(
            f"<tr><td>{h:02d}:00</td><td class=num>{dur(A['by_hour'].get(h, {}).get('median'))}</td>"
            f"<td class=num>{A['by_hour'].get(h, {}).get('n', 0)}</td>"
            + (f"<td class=num>{dur(Bp['by_hour'].get(h, {}).get('median'))}</td>"
               f"<td class=num>{Bp['by_hour'].get(h, {}).get('n', 0)}</td>" if Bp else "") + "</tr>" for h in hours)
        P.append("<details><summary>Таблица по часам</summary><table><tr><th>Час</th><th>Как было, обычно</th>"
                 "<th>посадок</th>" + ("<th>Стало, обычно</th><th>посадок</th>" if Bp else "") + f"</tr>{trs}</table>"
                 "</details>")
    # столы
    P.append("<h2>Столы, где чаще всего проблемы</h2>")
    for title, X in (("Как было", A), ("Стало", Bp)):
        if not X:
            continue
        pts = X["m"]["problem_tables"][:3]
        if not pts:
            P.append(f"<p>{title}: проблемных столов нет.</p>")
            continue
        per = {p["zone_id"]: p for p in X["m"]["per_table"]}
        items = "".join(
            f"<li><b>{e(table_label(p['zone_id']))}</b>: алертов {p['alerts']}, "
            f"обычно подход {dur(per[p['zone_id']]['avg_response_s'])} (в среднем), неубран всего "
            f"{dur(p['dirty_total_s'])}</li>" for p in pts)
        P.append(f"<p><b>{title}</b></p><ul>{items}</ul>")
    # оговорки
    P.append("<h2>Насколько этим цифрам можно верить</h2><ul>")
    P.append("<li>Цифры посчитаны автоматически по видео с ваших камер, без проверки человеком каждого случая.</li>")
    P.append("<li>Система не различает официанта и гостя «по виду» без настройки под ваш зал: перед отчётом "
             "проводится калибровка (персонал/гость) по записи вашего зала. Без неё время подхода недостоверно.</li>")
    P.append("<li>Столы, которые камеры не видят или видят наполовину (колонна, декор, темнота), в цифры не "
             "попадают или попадают с ошибками. Какие столы под наблюдением — указано в конфигурации пилота.</li>")
    P.append("<li>Точность на вашем зале проверяется ручной сверкой 30-минутного отрезка в неделю 1; "
             "до этой проверки воспринимайте цифры как оценку, а не как факт.</li>")
    lc = (Bp or A).get("low_conf")
    if lc:
        P.append(f"<li>{_pct(lc)} первых подходов засчитаны с пониженной уверенностью (человек у стола не "
                 "подтверждён как сотрудник).</li>")
    if synthetic:
        P.append("<li><b>Этот отчёт — пример на синтетических данных.</b></li>")
    P.append("</ul>")
    roi = A["m"]["roi"]
    P.append(f'<p class="small"><b>Как считаются деньги.</b> Часы пик — время, когда занято не меньше '
             f'{site.business.peak_occupancy * 100:.0f}% столов или у входа ждут гости. Для каждого стола берётся '
             f'время «гости ушли → стол убран», которое пришлось на часы пик, минус норматив уборки '
             f'{dur(site.business.target_cleanup_s)}. Сумма этих лишних минут делится на среднюю длительность визита '
             f'({dur(roi["avg_visit_s"])}{"" if roi["avg_visit_measured"] else ", из настроек"}) — это «потерянные '
             f'посадки» — и умножается на средний чек на стол ({fmt_money(site.business.avg_check_kzt, cur)}). Простой '
             f'вне часов пика не считается (в пустом зале грязный стол денег не стоит). «В месяц» — дневная цифра ×30. '
             'Медиана — значение, меньше которого половина случаев; 90-й перцентиль — меньше которого 90% случаев.</p>')
    P.append(f'<p class="small">Qonaq Vision · отчёт сформирован {datetime.now().strftime("%d.%m.%Y %H:%M")} · '
             f'событий в журнале: {A["events"]}{" и " + str(Bp["events"]) if Bp else ""}. Лица не распознаются, '
             'видео не сохраняется.</p>')
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Отчёт пилота — {e(site.name)}</title>'
            f'<style>{CSS}</style></head><body><div class="page">{"".join(P)}</div></body></html>')


def generate_pilot_report(site: SiteConfig, baseline: list[str], after: list[str] | None, out: str | Path,
                          b_from: str | None = None, b_to: str | None = None, a_from: str | None = None,
                          a_to: str | None = None) -> tuple[Path, dict, dict | None]:
    bs = load_period(baseline, b_from, b_to, site)
    A = analyze(bs, site)
    Bp = None
    if after:
        as_ = load_period(after, a_from, a_to, site)
        Bp = analyze(as_, site)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pilot_html(site, A, Bp), encoding="utf-8")
    return out, A, Bp


def summary_lines(A: dict, Bp: dict | None, site: SiteConfig) -> list[str]:
    cur = site.business.currency
    f = lambda X: (f"подход обычно {dur(X['median'])}, p90 {dur(X['p90'])}, дольше порога {_pct(X['share_over'])}, "  # noqa: E731
                   f"посадок/день {X['seatings_day']:.0f}, "
                   f"лишний простой в пик {X['dirty_peak_min_day']:.0f} мин/день, "
                   f"потери {fmt_money(X['loss_day'], cur)}/день")
    out = [f"Как было ({A['label']}): " + f(A)]
    if Bp:
        out.append(f"Стало ({Bp['label']}): " + f(Bp))
    return out

