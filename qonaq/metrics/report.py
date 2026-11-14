"""Отчёт смены: report.html (понятный не-технарю), report.md, telegram.txt (<= 10 строк). SPEC §8.2."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from qonaq.config import SiteConfig
from qonaq.events.store import EventStore
from qonaq.metrics.compute import compute_metrics
from qonaq.strings import fmt_money

ALERT_NAMES = {
    "no_waiter": "Гости долго ждут официанта",
    "dirty_table": "Стол долго не убран",
    "long_service_without_visit": "К гостям давно не подходили",
    "occupied_without_check": "Стол занят, а чек не открыт",
    "desk_unattended": "Очередь у стойки без сотрудника",
    "long_wait": "Посетитель долго ждёт",
}


def dur(seconds: float | None) -> str:
    """125 -> '2 мин 05 с'; None -> '—'."""
    if seconds is None:
        return "—"
    s = int(round(seconds))
    if s < 60:
        return f"{s} с"
    if s < 3600:
        return f"{s // 60} мин {s % 60:02d} с"
    return f"{s // 3600} ч {s % 3600 // 60:02d} мин"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{round(100 * x)}%"


def caveats(m: dict[str, Any]) -> list[str]:
    q, out = m["quality"], []
    out.append("Цифры получены автоматически по видео и не проверялись человеком на этой записи. "
               "Точность зависит от ракурса камеры; до пилота на вашем зале относитесь к ним как к оценке.")
    if q["preexisting_sessions"]:
        out.append(f"{q['preexisting_sessions']} стол(ов) были заняты уже в момент начала наблюдения — "
                   "для них время ожидания официанта неизвестно и в расчёт не входит.")
    low = m["response"].get("low_confidence_share")
    if low:
        out.append(f"В {pct(low)} случаев «подход официанта» определён без калибровки формы "
                   "(короткий визит стоящего человека к столу) — это менее надёжный признак. "
                   "После калибровки формы персонала (команда roles calibrate) точность выше.")
    if m["response"]["n"] < 10:
        out.append(f"Мало данных: измерено подходов — {m['response']['n']}. "
                   "Средние и медианы по такой выборке неустойчивы; нужен хотя бы один полный день.")
    if not m["roi"]["avg_visit_measured"]:
        out.append(f"Средняя длительность визита взята из настроек ({dur(m['roi']['avg_visit_s'])}), "
                   "так как завершённых визитов в записи меньше пяти.")
    out.append("Потери считаются консервативно: только лишний простой неубранного стола в пиковое время "
               f"(сверх норматива {dur(m['roi']['target_cleanup_s'])}) и только когда зал загружен "
               "или у входа очередь. В пустом зале грязный стол денег не стоит.")
    out.append("Система не распознаёт лица и не идентифицирует людей; видео не сохраняется, "
               "в отчёт попадают только агрегированные цифры.")
    return out


def telegram_text(m: dict[str, Any], currency: str = "₸") -> str:
    r, d, roi, t = m["response"], m["dirty"], m["roi"], m["turnover"]
    lines = [f"📊 {m['site']['name']} · {m['period']['from']}–{m['period']['to']}"]
    if m.get("clinic"):
        c = m["clinic"]
        lines += [
            f"Ожидание: медиана {dur(c['waits']['median'])}, p90 {dur(c['waits']['p90'])} ({c['waits']['n']} чел.)",
            f"Обслужено: {c['served']}, ушли не дождавшись: {c['left_without_service']}",
            f"Обслуживание у стойки: в среднем {dur(c['service']['mean'])}",
        ]
        if c["desk_unattended_alerts"]:
            lines.append(f"⚠ Стойка пустовала при очереди: {c['desk_unattended_alerts']} раз")
        return "\n".join(lines[:10])
    lines.append(f"Посадок: {t['seatings']} · загрузка зала {pct(m['tables']['avg_load'])}")
    if r["n"]:
        lines.append(f"Подход официанта: медиана {dur(r['median'])}, p90 {dur(r['p90'])}")
        lines.append(f"Ждали дольше {dur(r['threshold_s'])}: {pct(r['share_over_threshold'])} столов")
    else:
        lines.append("Подход официанта: нет измерений за период")
    if m["walkouts"]["n"]:
        lines.append(f"🚪 Ушли, не дождавшись: {m['walkouts']['n']}")
    if d["n"]:
        lines.append(f"Неубранные столы: в среднем {dur(d['mean'])}, максимум {dur(d['max'])}")
    if roi["loss_kzt"] or roi["walkout_loss_kzt"]:
        lines.append(f"💸 Оценка потерь: {fmt_money(roi['loss_kzt'] + roi['walkout_loss_kzt'], currency)}")
    if m["problem_tables"]:
        lines.append("Проблемные столы: " + ", ".join(p["zone_id"] for p in m["problem_tables"][:3]))
    total_alerts = sum(m["alerts"].values())
    if total_alerts:
        lines.append(f"⚠ Алертов за период: {total_alerts}")
    lines.append("Оценка по видео, без проверки человеком.")
    return "\n".join(lines[:10])


def shift_summary_text(m: dict[str, Any], currency: str = "₸", day_label: str = "") -> str:
    """Сводка смены владельцу заведения в Telegram (S4): человеческими словами, без «медиан» и «p90»."""
    from qonaq.strings import human_dur, plural, table_label

    r, d, roi, t = m["response"], m["dirty"], m["roi"], m["turnover"]
    head = f"Qonaq · итоги дня · {m['site']['name']}" + (f" · {day_label}" if day_label else "")
    lines = [head]
    if m.get("clinic"):
        c = m["clinic"]
        lines += [f"Посетителей дождались приёма: {c['served']}, ушли не дождавшись: {c['left_without_service']}",
                  f"Ожидание: обычно {dur(c['waits']['median'])}, у худших 10% — дольше {dur(c['waits']['p90'])}"]
        lines.append("Это оценка по видео без проверки человеком.")
        return "\n".join(lines)
    lines.append(f"Посадок за столы: {t['seatings']}")
    if r["n"]:
        lines.append(f"Официант подходил обычно за {dur(r['median'])}; у худших 10% столов — дольше {dur(r['p90'])}")
        if r.get("share_over_threshold") is not None:
            total = r["n"] + r.get("unserved_over_threshold", 0)
            over = round(r["share_over_threshold"] * total)
            lines.append(f"Ждали официанта дольше {human_dur(r['threshold_s'])}: "
                         f"{plural(over, 'стол', 'стола', 'столов')} из {total} ({pct(r['share_over_threshold'])})")
    else:
        lines.append("Подход официанта: за день не измерен (мало посадок или камера не видела)")
    if m["walkouts"]["n"]:
        lines.append(f"Ушли, не дождавшись официанта: {m['walkouts']['n']}")
    if roi["peak_s"] > 0 and roi["excess_dirty_in_peak_s"] > 0:
        lines.append(f"Неубранные столы в часы пик: лишних {dur(roi['excess_dirty_in_peak_s'])} в сумме")
    elif d["n"]:
        lines.append(f"Столы убирали в среднем за {dur(d['mean'])}")
    loss = roi["loss_kzt"] + roi["walkout_loss_kzt"]
    if loss:
        lines.append(f"Оценка упущенной выручки: около {fmt_money(loss, currency)} (только часы пик и ушедшие)")
    if m["problem_tables"]:
        lines.append("Чаще всего проблемы: " + ", ".join(table_label(p["zone_id"]) for p in m["problem_tables"][:3]))
    lines.append("Это оценка по видео без проверки человеком.")
    return "\n".join(lines)


def markdown_report(m: dict[str, Any], currency: str = "₸") -> str:
    r, d, roi, t, v = m["response"], m["dirty"], m["roi"], m["turnover"], m["visit"]
    L = [f"# Отчёт смены — {m['site']['name']}", "",
         f"Период: {m['period']['from']} – {m['period']['to']} ({dur(m['period']['duration_s'])}). "
         f"Столов под наблюдением: {m['tables']['total']}.", ""]
    if m.get("clinic"):
        c = m["clinic"]
        L += ["## Ожидание", "", "| Показатель | Значение |", "|---|---|",
              f"| Посетителей (завершённых ожиданий) | {c['waits']['n']} |",
              f"| Среднее ожидание | {dur(c['waits']['mean'])} |",
              f"| Медиана | {dur(c['waits']['median'])} |", f"| p90 | {dur(c['waits']['p90'])} |",
              f"| Перешли к стойке | {c['served']} |",
              f"| Ушли, не дождавшись | {c['left_without_service']} |",
              f"| Среднее время у стойки | {dur(c['service']['mean'])} |",
              f"| Стойка пустовала при очереди (алерты) | {c['desk_unattended_alerts']} |", ""]
    else:
        L += ["## Главное", "", "| Показатель | Значение |", "|---|---|",
              f"| Посадок за период | {t['seatings']} (новых: {t['new_seatings']}) |",
              f"| Средняя загрузка зала | {pct(m['tables']['avg_load'])} |",
              f"| Время до подхода официанта — медиана | {dur(r['median'])} |",
              f"| — среднее / p90 | {dur(r['mean'])} / {dur(r['p90'])} |",
              f"| Доля ожиданий дольше {dur(r['threshold_s'])} | {pct(r['share_over_threshold'])} |",
              f"| Измерено подходов | {r['n']} |",
              f"| Ушли до обслуживания | {m['walkouts']['n']} |",
              f"| Сейчас ждут официанта | {m['still_waiting']} |",
              f"| Неубранный стол — среднее / максимум | {dur(d['mean'])} / {dur(d['max'])} |",
              f"| Сели за неубранный стол | {d['seated_at_dirty']} |",
              f"| Средняя длительность визита | {dur(v['mean'])} (визитов: {v['n']}) |",
              f"| Подходов персонала на стол | {m['visits_per_session']['mean'] if m['visits_per_session']['n'] else '—'} |",
              f"| Оборачиваемость (посадок на стол) | {t['per_table'] if t['per_table'] is not None else '—'} |",
              ""]
        L += ["## Потери в деньгах (консервативная оценка)", "",
              f"- Пиковое время (зал загружен ≥ порога или очередь у входа): {dur(roi['peak_s'])} "
              f"({pct(roi['peak_share'])} периода).",
              f"- Лишний простой неубранных столов в пик: {dur(roi['excess_dirty_in_peak_s'])} "
              f"→ потерянных посадок: {roi['lost_seatings']}.",
              f"- **Потери от неубранных столов: {fmt_money(roi['loss_kzt'], currency)}** "
              f"(средний чек на стол {fmt_money(roi['avg_check_kzt'], currency)}).",
              f"- Отдельно — гости ушли, не дождавшись официанта: {roi['walkouts']} "
              f"(до {fmt_money(roi['walkout_loss_kzt'], currency)}).",
              f"- Свободные столы в пик (суммарно): {dur(roi['free_table_s_in_peak'])}.", ""]
        if m["problem_tables"]:
            L += ["## Проблемные столы", "", "| Стол | Посадок | Ср. ожидание | Макс. ожидание | Не убран (всего) | Алертов |",
                  "|---|---|---|---|---|---|"]
            for p in m["problem_tables"]:
                L.append(f"| {p['zone_id']} | {p['sessions']} | {dur(p['avg_response_s'])} | "
                         f"{dur(p['max_response_s'])} | {dur(p['dirty_total_s'])} | {p['alerts']} |")
            L.append("")
    if m["alerts"]:
        L += ["## Алерты", ""] + [f"- {ALERT_NAMES.get(k, k)}: {n}" for k, n in sorted(m["alerts"].items())] + [""]
    L += ["## Оговорки о точности", ""] + [f"- {c}" for c in caveats(m)] + [""]
    return "\n".join(L)


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mut:#667085;--line:#e4e7ec;--acc:#2563eb;--bad:#d92d20;--ok:#079455;--warn:#dc6803}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b21;--ink:#eceef2;--mut:#98a2b3;--line:#2a2f38;--acc:#6ea8fe;--bad:#f97066;--ok:#47cd89;--warn:#fdb022}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 12px}
.sub{color:var(--mut);margin:0 0 20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.k{color:var(--mut);font-size:13px}.v{font-size:28px;font-weight:700;line-height:1.2;margin-top:4px;font-variant-numeric:tabular-nums}
.n{color:var(--mut);font-size:13px;margin-top:4px}.bad .v{color:var(--bad)}.ok .v{color:var(--ok)}.warn .v{color:var(--warn)}
.money{border-left:4px solid var(--bad)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);font-size:14px}th{color:var(--mut);font-weight:600}
.scroll{overflow-x:auto}.heat td{text-align:center;min-width:44px;font-size:12px;border:2px solid var(--card)}
.heat th{font-size:12px;text-align:center}.heat th:first-child,.heat td:first-child{text-align:left;font-weight:600}
ul{margin:0;padding-left:20px}li{margin:6px 0;color:var(--mut)}li b{color:var(--ink)}
.foot{color:var(--mut);font-size:13px;margin-top:32px}
"""


def _card(label: str, value: str, note: str = "", cls: str = "") -> str:
    return (f'<div class="card {cls}"><div class="k">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(value)}</div><div class="n">{html.escape(note)}</div></div>')


def _heat_cell(v: float) -> str:
    alpha = 0.08 + 0.82 * max(0.0, min(1.0, v))
    ink = "#fff" if v > 0.55 else "inherit"
    return f'<td style="background:rgba(37,99,235,{alpha:.2f});color:{ink}">{round(100 * v)}</td>'


def html_report(m: dict[str, Any], currency: str = "₸") -> str:
    r, d, roi, t, v = m["response"], m["dirty"], m["roi"], m["turnover"], m["visit"]
    e = html.escape
    parts = [f"<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Отчёт смены — {e(m['site']['name'])}</title><style>{CSS}</style></head><body><main>",
             f"<h1>Отчёт смены — {e(m['site']['name'])}</h1>",
             f"<p class='sub'>Период наблюдения: {e(m['period']['from'])} – {e(m['period']['to'])} "
             f"({dur(m['period']['duration_s'])}) · зон-столов: {m['tables']['total']}</p>"]
    if m.get("clinic"):
        c = m["clinic"]
        parts.append("<div class='grid'>" + "".join([
            _card("Ожидание — медиана", dur(c["waits"]["median"]), f"завершённых ожиданий: {c['waits']['n']}"),
            _card("Ожидание — 9 из 10 ждут не дольше", dur(c["waits"]["p90"]), "p90"),
            _card("Перешли к стойке", str(c["served"]), "дождались своей очереди", "ok"),
            _card("Ушли, не дождавшись", str(c["left_without_service"]), "",
                  "bad" if c["left_without_service"] else ""),
            _card("Время у стойки — среднее", dur(c["service"]["mean"]), f"обслуживаний: {c['service']['n']}"),
            _card("Стойка пустовала при очереди", str(c["desk_unattended_alerts"]), "алертов",
                  "warn" if c["desk_unattended_alerts"] else ""),
        ]) + "</div>")
    else:
        over = r["share_over_threshold"]
        parts.append("<div class='grid'>" + "".join([
            _card("Гости ждут официанта (медиана)", dur(r["median"]),
                  f"измерено подходов: {r['n']}" if r["n"] else "за период нет измерений",
                  "bad" if (r["median"] or 0) > r["threshold_s"] else ""),
            _card(f"Ждали дольше {dur(r['threshold_s'])}", pct(over), "доля столов с долгим ожиданием",
                  "bad" if (over or 0) > 0.2 else ""),
            _card("Ушли, не дождавшись", str(m["walkouts"]["n"]), "сели и ушли до подхода официанта",
                  "bad" if m["walkouts"]["n"] else "ok"),
            _card("Стол не убран (в среднем)", dur(d["mean"]), f"максимум {dur(d['max'])}",
                  "warn" if (d["mean"] or 0) > 300 else ""),
            _card("Посадок за период", str(t["seatings"]), f"на стол: {t['per_table'] if t['per_table'] is not None else '—'}"),
            _card("Средняя загрузка зала", pct(m["tables"]["avg_load"]), f"пик: {dur(roi['peak_s'])}"),
        ]) + "</div>")
        parts.append("<h2>Сколько денег это стоит</h2><div class='grid'>" + "".join([
            _card("Потери от неубранных столов в пик", fmt_money(roi["loss_kzt"], currency),
                  f"потерянных посадок: {roi['lost_seatings']} × средний чек {fmt_money(roi['avg_check_kzt'], currency)}",
                  "money"),
            _card("Гости ушли до обслуживания", fmt_money(roi["walkout_loss_kzt"], currency),
                  f"столов: {roi['walkouts']} (считается отдельно)", "money" if roi["walkouts"] else ""),
            _card("Лишний простой в пик", dur(roi["excess_dirty_in_peak_s"]),
                  f"сверх норматива уборки {dur(roi['target_cleanup_s'])}"),
        ]) + "</div>")
        parts.append("<h2>Подробно</h2><div class='card scroll'><table>" + "".join(
            f"<tr><td>{e(a)}</td><td><b>{e(b)}</b></td></tr>" for a, b in [
                ("Время до подхода — среднее / медиана / p90", f"{dur(r['mean'])} / {dur(r['median'])} / {dur(r['p90'])}"),
                ("Сейчас ждут официанта", str(m["still_waiting"])),
                ("Средняя длительность визита", f"{dur(v['mean'])} (завершённых визитов: {v['n']})"),
                ("Подходов персонала на стол (в среднем)",
                 str(m["visits_per_session"]["mean"]) if m["visits_per_session"]["n"] else "—"),
                ("Сели за неубранный стол", str(d["seated_at_dirty"])),
                ("Столы не убраны на конец периода", str(d["open"])),
                ("Свободные столы в пиковое время (суммарно)", dur(roi["free_table_s_in_peak"])),
                ("Очередь у входа", f"максимум {m['queue']['max']} чел., с очередью {dur(m['queue']['time_with_queue_s'])}"
                 if m.get("queue") else "зона входа не размечена или очереди не было"),
            ]) + "</table></div>")
        if m["problem_tables"]:
            parts.append("<h2>Столы, которым нужно внимание</h2><div class='card scroll'><table><tr><th>Стол</th>"
                         "<th>Посадок</th><th>Ср. ожидание</th><th>Макс. ожидание</th><th>Не убран (всего)</th>"
                         "<th>Ушли</th><th>Алертов</th></tr>")
            for p in m["problem_tables"]:
                parts.append(f"<tr><td><b>{e(p['zone_id'])}</b></td><td>{p['sessions']}</td>"
                             f"<td>{dur(p['avg_response_s'])}</td><td>{dur(p['max_response_s'])}</td>"
                             f"<td>{dur(p['dirty_total_s'])}</td><td>{p['walkouts']}</td><td>{p['alerts']}</td></tr>")
            parts.append("</table></div>")
        h = m["heatmap"]
        if h["rows"] and h["labels"]:
            parts.append(f"<h2>Загрузка зала по времени</h2><p class='sub'>Процент времени, когда стол был занят "
                         f"(шаг {dur(h['bucket_s'])}).</p><div class='card scroll'><table class='heat'><tr><th>Стол</th>"
                         + "".join(f"<th>{e(x)}</th>" for x in h["labels"]) + "</tr>")
            for row in h["rows"]:
                parts.append(f"<tr><td>{e(row['zone_id'])}</td>" + "".join(_heat_cell(x) for x in row["values"]) + "</tr>")
            parts.append("<tr><td>Весь зал</td>" + "".join(_heat_cell(x) for x in h["total"]) + "</tr></table></div>")
    if m["alerts"]:
        parts.append("<h2>Алерты за период</h2><div class='card'><table>" + "".join(
            f"<tr><td>{e(ALERT_NAMES.get(k, k))}</td><td><b>{n}</b></td></tr>" for k, n in sorted(m["alerts"].items()))
            + "</table></div>")
    parts.append("<h2>Насколько этим цифрам можно верить</h2><div class='card'><ul>"
                 + "".join(f"<li>{e(c)}</li>" for c in caveats(m)) + "</ul></div>")
    parts.append("<p class='foot'>Qonaq Vision · отчёт построен только по журналу событий; "
                 "видео не сохраняется, лица не распознаются.</p></main></body></html>")
    return "".join(parts)


def generate_report(run_dir: str | Path, site: SiteConfig | None = None) -> dict[str, Path]:
    run_dir = Path(run_dir)
    db = run_dir / "events.sqlite"
    if not db.exists():
        raise FileNotFoundError(f"нет журнала событий: {db}")
    site = site or SiteConfig()
    zones = None
    run_json = run_dir / "run.json"
    if run_json.exists():
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        if not site.cameras:
            site.site_id, site.name = meta.get("site_id", site.site_id), meta.get("name", site.name)
            site.profile = meta.get("profile", site.profile)
        zones = [z for zs in meta.get("zones", {}).values() for z in zs if z.startswith("T")]
    store = EventStore(db)
    try:
        m = compute_metrics(store, site, zones=zones)
    finally:
        store.close()
    cur = site.business.currency
    out = {"html": run_dir / "report.html", "md": run_dir / "report.md", "telegram": run_dir / "telegram.txt",
           "json": run_dir / "metrics.json"}
    out["html"].write_text(html_report(m, cur), encoding="utf-8")
    out["md"].write_text(markdown_report(m, cur), encoding="utf-8")
    out["telegram"].write_text(telegram_text(m, cur), encoding="utf-8")
    out["json"].write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
