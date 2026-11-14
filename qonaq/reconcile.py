"""Сверка «занято по видео» с учётом (SPRINT3 B3): оплаченные сессии / чеки из CSV против сессий по видео.

Общий модуль для двух профилей:
- billiard — сессии игры (game_started/game_ended) против оплаченных сессий стола (стол, начало, конец, сумма):
  «игра без счётчика» (по видео играли, оплаченной сессии нет), «играли дольше оплаченного» (минуты до начала /
  после конца оплаченной сессии), «счётчик шёл, стол пустой» (в пользу гостя);
- restaurant — сессии стола (посадка → уход) против чеков (стол, открыт, закрыт, сумма): «стол занят, чека нет».

Допуски: рассинхрон часов видео и кассы (tolerance_s), округление в учёте (rounding_min), мелочь не показываем
(min_minutes). Сверка только внутри периода, когда видео было (coverage); остальное — «вне периода записи».
Тексты нейтральные, без людей: «Стол 4, 21:10–22:40, по учёту не оплачено».
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from html import escape
from pathlib import Path
from typing import Any

import yaml

Interval = tuple[float, float]

HEADERS = {
    "table": ["стол", "table", "№ стола", "номер стола", "table_id", "ресурс", "место", "объект", "tableid", "№"],
    "start": ["начало", "start", "открыт", "время начала", "start_time", "opened", "открытие", "начало игры", "с"],
    "end": ["конец", "end", "закрыт", "время окончания", "end_time", "closed", "закрытие", "окончание", "по"],
    "duration": ["длительность", "duration", "минут", "время игры", "мин", "продолжительность"],
    "amount": ["сумма", "amount", "итого", "стоимость", "total", "оплачено", "к оплате"],
    "date": ["дата", "date", "день"],
}
DT_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%d.%m.%Y %H:%M:%S",
              "%d.%m.%Y %H:%M", "%d.%m.%y %H:%M", "%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M"]
TIME_FORMATS = ["%H:%M:%S", "%H:%M"]
DATE_FORMATS = ["%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y"]


def local_tz(name: str = "Asia/Almaty") -> tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — Windows без пакета tzdata: РК — UTC+5 круглый год (с 2024)
        return timezone(timedelta(hours=5))


# ---------------------------------------------------------------------------------------------- CSV
@dataclass
class Paid:
    table: str
    start: float  # epoch
    end: float
    amount: float | None = None
    row: int = 0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower().replace("ё", "е"))


def _find_cols(header: list[str]) -> dict[str, int]:
    h = [_norm(x) for x in header]
    out: dict[str, int] = {}
    for key, names in HEADERS.items():
        for i, col in enumerate(h):
            if col in names and i not in out.values():
                out[key] = i
                break
        else:
            for i, col in enumerate(h):  # «Время начала игры», «Сумма, ₸»
                if any(col.startswith(n) for n in names if len(n) > 2) and i not in out.values():
                    out[key] = i
                    break
    return out


def _amount(s: str) -> float | None:
    s = re.sub(r"[^\d,.\-]", "", s or "")
    if not s:
        return None
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _duration_s(s: str) -> float | None:
    s = (s or "").strip()
    m = re.fullmatch(r"(\d+):(\d{2})(?::(\d{2}))?", s)
    if m:
        return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3] or 0)
    v = _amount(s)
    return v * 60 if v is not None else None  # число — минуты


def parse_dt(s: str, day: date | None, tz: tzinfo, day_start_h: int = 6) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    for f in DT_FORMATS:
        try:
            return datetime.strptime(s, f).replace(tzinfo=tz)
        except ValueError:
            pass
    for f in TIME_FORMATS:
        try:
            t = datetime.strptime(s, f).time()
        except ValueError:
            continue
        if day is None:
            raise ValueError(f"время без даты «{s}»: укажите --date (рабочий день)")
        d = day + timedelta(days=1) if t.hour < day_start_h else day  # смена после полуночи — следующий день
        return datetime.combine(d, t, tz)
    raise ValueError(f"не понимаю время «{s}»")


def _parse_date(s: str) -> date | None:
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(s.strip(), f).date()
        except ValueError:
            pass
    return None


def read_paid_csv(path: str | Path, day: date | None = None, tz: tzinfo | None = None,
                  day_start_h: int = 6) -> tuple[list[Paid], list[str]]:
    """CSV/TSV из программы учёта: разделитель , ; или таб; кодировка UTF-8 (с BOM) или cp1251 (Excel)."""
    tz = tz or local_tz()
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:4096]
    delim = max([";", ",", "\t"], key=lambda d: sample.splitlines()[0].count(d) if sample else 0)
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return [], ["пустой файл"]
    cols = _find_cols(rows[0])
    warn: list[str] = []
    missing = [k for k in ("table", "start") if k not in cols]
    if missing or ("end" not in cols and "duration" not in cols):
        raise ValueError(f"не нашёл колонки {missing or ['конец/длительность']} в заголовке: {rows[0]}")
    out: list[Paid] = []
    for n, r in enumerate(rows[1:], start=2):
        get = lambda k, r=r: r[cols[k]].strip() if k in cols and cols[k] < len(r) else ""  # noqa: E731
        try:
            d = _parse_date(get("date")) if "date" in cols else None
            st = parse_dt(get("start"), d or day, tz, day_start_h)
            if st is None:
                warn.append(f"строка {n}: нет начала — пропущена")
                continue
            en = parse_dt(get("end"), d or day, tz, day_start_h) if get("end") else None
            if en is None:
                dur = _duration_s(get("duration"))
                if dur is None:
                    warn.append(f"строка {n}: нет конца и длительности — пропущена")
                    continue
                en = st + timedelta(seconds=dur)
            if en <= st:  # 23:30 → 01:10 без даты
                en += timedelta(days=1)
            out.append(Paid(get("table"), st.timestamp(), en.timestamp(), _amount(get("amount")), n))
        except ValueError as e:
            warn.append(f"строка {n}: {e}")
    return out, warn


# ---------------------------------------------------------------------------------------------- интервалы
def merge(iv: list[Interval]) -> list[Interval]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def subtract(x: Interval, ys: list[Interval]) -> list[Interval]:
    parts = [x]
    for c, d in merge(ys):
        nxt = []
        for a, b in parts:
            if d <= a or c >= b:
                nxt.append((a, b))
                continue
            if c > a:
                nxt.append((a, c))
            if d < b:
                nxt.append((d, b))
        parts = nxt
    return parts


def overlap(x: Interval, ys: list[Interval]) -> float:
    return sum(max(0.0, min(x[1], d) - max(x[0], c)) for c, d in ys)


# ---------------------------------------------------------------------------------------------- сверка
@dataclass
class Settings:
    profile: str = "billiard"
    tolerance_s: float = 120.0  # рассинхрон часов кассы и видео
    rounding_min: float = 5.0  # учёт округляет сессию (вверх) до N минут
    min_minutes: float = 10.0  # отрезки короче не показываем
    idle_extra_s: float = 300.0  # «стол пустой»: к концу игры по видео добавляем (сдача кия, расчёт)
    tariff_per_hour: float = 2500.0
    tariffs: dict[str, float] = field(default_factory=dict)  # по зоне
    avg_check: float = 15000.0  # ресторан: оценка «чека нет» — средний чек
    currency: str = "₸"
    timezone: str = "Asia/Almaty"
    table_map: dict[str, dict[str, Any]] = field(default_factory=dict)  # «Стол 4» -> {zone, camera, tariff, name}

    def tariff(self, zone: str) -> float:
        for m in self.table_map.values():
            if m.get("zone") == zone and m.get("tariff"):
                return float(m["tariff"])
        return float(self.tariffs.get(zone, self.tariff_per_hour))


@dataclass
class Finding:
    kind: str  # unpaid_play | overtime | idle_paid | no_check
    zone: str
    table: str  # как в учёте / на панели
    start: float
    end: float
    minutes: float
    money: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


KIND_TITLE = {
    "unpaid_play": "Игра без счётчика",
    "overtime": "Играли дольше оплаченного",
    "idle_paid": "Счётчик шёл, стол пустой (в пользу гостя)",
    "no_check": "Стол занят, чека нет",
}


def zone_for(pos_table: str, zones: set[str], table_map: dict[str, dict[str, Any]]) -> str | None:
    """«Стол 4» → B4 / T04 / T4 (или по карте --map)."""
    m = table_map.get(pos_table) or table_map.get(_norm(pos_table))
    if m:
        return m.get("zone")
    if pos_table in zones:
        return pos_table
    digits = re.findall(r"\d+", pos_table)
    if digits:
        n = int(digits[-1])
        for cand in (f"B{n}", f"T{n:02d}", f"T{n}", f"B{n:02d}"):
            if cand in zones:
                return cand
    return None


def table_title(zone: str, table_map: dict[str, dict[str, Any]]) -> str:
    for k, m in table_map.items():
        if m.get("zone") == zone:
            return m.get("name") or k
    d = re.findall(r"\d+", zone)
    return f"Стол {int(d[-1])}" if d else zone


def reconcile(video: dict[str, list[Interval]], paid: list[Paid], cfg: Settings,
              coverage: list[Interval] | None = None, known_zones: set[str] | None = None) -> dict[str, Any]:
    """coverage — когда система видела (None — всегда). Вне него ничего не утверждаем.
    known_zones — все столы на камерах (в т.ч. без игры за период), чтобы «пустой стол» тоже сверялся."""
    zones = set(video) | set(known_zones or ()) | {m.get("zone") for m in cfg.table_map.values() if m.get("zone")}
    by_zone: dict[str, list[Paid]] = {}
    unknown: list[str] = []
    outside = 0
    for p in paid:
        z = zone_for(p.table, zones, cfg.table_map)
        if z is None:
            unknown.append(p.table)
            continue
        if coverage is not None and overlap((p.start, p.end), coverage) <= 0:
            outside += 1
            continue
        by_zone.setdefault(z, []).append(p)
    tol = cfg.tolerance_s
    rnd = cfg.rounding_min * 60
    min_s = cfg.min_minutes * 60
    findings: list[Finding] = []
    tables = []
    for z in sorted(zones):
        rate = cfg.tariff(z) / 3600.0
        title = table_title(z, cfg.table_map) if cfg.table_map else table_title(z, {})
        v = merge(video.get(z, []))
        if coverage is not None:
            v = merge([(max(a, c), min(b, d)) for a, b in v for c, d in coverage if min(b, d) > max(a, c)])
        ps = by_zone.get(z, [])
        p_raw = merge([(p.start, p.end) for p in ps])
        p_exp = merge([(a - tol, b + tol + rnd) for a, b in p_raw])
        for a, b in v:
            touched = overlap((a, b), p_exp) > 0
            if cfg.profile == "restaurant":
                if not touched and b - a >= min_s:
                    findings.append(Finding("no_check", z, title, a, b, (b - a) / 60, cfg.avg_check,
                                            "по учёту чека нет"))
                continue
            if not touched:
                if b - a >= min_s:
                    findings.append(Finding("unpaid_play", z, title, a, b, (b - a) / 60, (b - a) * rate,
                                            "по учёту не оплачено"))
                continue
            for c, d in subtract((a, b), p_exp):
                if d - c >= min_s:
                    findings.append(Finding("overtime", z, title, c, d, (d - c) / 60, (d - c) * rate,
                                            "по видео игра шла вне оплаченного времени"))
        if cfg.profile == "billiard":
            v_exp = merge([(a - tol, b + tol + cfg.idle_extra_s) for a, b in v])
            blind = [] if coverage is None else subtract((min(p.start for p in ps) - 1, max(p.end for p in ps) + 1),
                                                         coverage) if ps else []
            for p in ps:
                for c, d in subtract((p.start, p.end), v_exp + blind):  # вслепую — ничего не утверждаем
                    if d - c >= min_s:
                        findings.append(Finding("idle_paid", z, title, c, d, (d - c) / 60, (d - c) * rate,
                                                "по учёту счётчик шёл, по видео игры нет"))
        tables.append({"zone": z, "title": title, "video_min": round(sum(b - a for a, b in v) / 60, 1),
                       "paid_min": round(sum(b - a for a, b in p_raw) / 60, 1), "paid_sessions": len(ps),
                       "video_sessions": len(v), "paid_amount": round(sum(p.amount or 0 for p in ps), 0),
                       "tariff": cfg.tariff(z)})
    tot: dict[str, dict[str, float]] = {}
    for f in findings:
        t = tot.setdefault(f.kind, {"count": 0, "minutes": 0.0, "money": 0.0})
        t["count"] += 1
        t["minutes"] += f.minutes
        t["money"] += f.money
    return {"findings": [f.to_dict() for f in sorted(findings, key=lambda f: (f.start, f.zone))],
            "totals": {k: {"count": v["count"], "minutes": round(v["minutes"], 1), "money": round(v["money"])}
                       for k, v in tot.items()},
            "tables": tables, "unknown_tables": sorted(set(unknown)), "outside_coverage": outside,
            "coverage": [list(x) for x in coverage] if coverage is not None else None, "settings": {k: v for k, v in asdict(cfg).items() if k != "table_map"},
            "profile": cfg.profile}


# ---------------------------------------------------------------------------------------------- вход
def video_sessions(store, profile: str, offset: float = 0.0) -> dict[str, list[Interval]]:
    """Интервалы «занято по видео» из events.sqlite (ts + offset → epoch)."""
    if profile == "billiard":
        from qonaq.billiard.evaluate import sessions_from_events

        evs = store.events(types=["game_started", "game_ended"])
        return {z: [(a + offset, b + offset) for a, b in iv] for z, iv in sessions_from_events(evs).items()}
    out: dict[str, list[Interval]] = {}
    for s in store.sessions("table"):
        a = s.get("seated_ts")
        b = s.get("left_ts") or s.get("freed_ts")
        if a is None or b is None:
            continue
        out.setdefault(s["zone_id"], []).append((float(a) + offset, float(b) + offset))
    return out


def coverage_from_store(store, offset: float = 0.0) -> dict[str, list[Interval]]:
    """Когда система реально видела: по событиям heartbeat (live пишет раз в минуту: окно и камеры с кадрами).
    Ключ "*" — хоть одна камера; иначе id камеры. Без heartbeat (прогон файла, старые журналы) — один интервал
    от начала до конца журнала."""
    hb = store.events(types=["heartbeat"])
    out: dict[str, list[Interval]] = {}
    if hb:
        for e in hb:
            p = e.get("payload") or {}
            w = float(p.get("window_s", 60.0)) * 1.5  # с запасом: окна стыкуются
            iv = (e["ts"] - w + offset, e["ts"] + offset)
            up = p.get("up") or []
            for c in up:
                out.setdefault(c, []).append(iv)
            if up:
                out.setdefault("*", []).append(iv)
        return {k: merge(v) for k, v in out.items()}
    meta = store.meta()
    evs = store.events(limit=None)
    if not evs and not meta:
        return {}
    a = min([float(meta[k]) for k in ("t_start", "started_wall") if k in meta and meta[k] is not None] +
            [e["ts"] for e in evs[:1]] or [0.0])
    b = max([float(meta[k]) for k in ("t_end",) if k in meta and meta[k] is not None] +
            ([e["ts"] for e in evs[-1:]]) + ([a + float(meta["scene_s"])] if meta.get("scene_s") else []))
    return {"*": [(a + offset, b + offset)]}


def load_map(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------------------------- вывод
def _hm(ts: float, tz: tzinfo) -> str:
    return datetime.fromtimestamp(ts + 30, tz).strftime("%H:%M")  # до ближайшей минуты


def _dur(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60} ч {m % 60:02d} мин" if m >= 60 else f"{m} мин"


def _money(x: float, cur: str) -> str:
    return f"{int(round(x)):,}".replace(",", " ") + f" {cur}"


def finding_line(f: dict[str, Any], tz: tzinfo, cur: str) -> str:
    money = f" ≈ {_money(f['money'], cur)}" if f["kind"] != "no_check" else f" (средний чек ≈ {_money(f['money'], cur)})"
    return f"{f['table']}, {_hm(f['start'], tz)}–{_hm(f['end'], tz)} ({_dur(f['minutes'])}) — {f['note']}{money}"


def telegram_text(res: dict[str, Any], title: str = "", tz: tzinfo | None = None, max_lines: int = 5) -> str:
    tz = tz or local_tz(res["settings"].get("timezone", "Asia/Almaty"))
    cur = res["settings"].get("currency", "₸")
    lines = [f"Сверка с учётом{(' — ' + title) if title else ''}"]
    if not res["findings"]:
        lines.append("Расхождений больше допуска нет.")
    for kind in ("unpaid_play", "no_check", "overtime", "idle_paid"):
        t = res["totals"].get(kind)
        if not t:
            continue
        lines.append(f"• {KIND_TITLE[kind]}: {t['count']}, {_dur(t['minutes'])} ≈ {_money(t['money'], cur)}")
        fs = [f for f in res["findings"] if f["kind"] == kind][:max_lines]
        lines += ["   " + finding_line(f, tz, cur) for f in fs]
    if res["unknown_tables"]:
        lines.append(f"Столы из учёта без зоны на камере: {', '.join(res['unknown_tables'])}")
    lines.append("Это расхождение видео и учёта, а не вывод о сотрудниках: проверьте по записи и кассе.")
    return "\n".join(lines)


def html_report(res: dict[str, Any], title: str = "", tz: tzinfo | None = None) -> str:
    tz = tz or local_tz(res["settings"].get("timezone", "Asia/Almaty"))
    cur = res["settings"].get("currency", "₸")
    s = res["settings"]
    cov = res.get("coverage")
    cov_txt = (f"{datetime.fromtimestamp(cov[0][0], tz):%d.%m %H:%M} – {datetime.fromtimestamp(cov[-1][1], tz):%d.%m %H:%M}"
               + (f" (перерывов в работе системы: {len(cov) - 1})" if len(cov) > 1 else "")
               if cov else "весь журнал")
    cards = []
    for kind in ("unpaid_play", "no_check", "overtime", "idle_paid"):
        t = res["totals"].get(kind)
        if t is None and kind in ("no_check",) and res["profile"] != "restaurant":
            continue
        if t is None and kind in ("unpaid_play", "overtime", "idle_paid") and res["profile"] == "restaurant":
            continue
        t = t or {"count": 0, "minutes": 0, "money": 0}
        good = kind == "idle_paid"
        cards.append(f'<div class="card{" guest" if good else ""}"><div class="k">{KIND_TITLE[kind]}</div>'
                     f'<div class="v">{_dur(t["minutes"])}</div><div class="m">≈ {_money(t["money"], cur)}'
                     f' · {t["count"]} шт.</div></div>')
    rows = "".join(
        f"<tr><td>{escape(f['table'])}</td><td>{KIND_TITLE[f['kind']]}</td>"
        f"<td>{_hm(f['start'], tz)}–{_hm(f['end'], tz)}</td><td>{_dur(f['minutes'])}</td>"
        f"<td>{_money(f['money'], cur)}</td><td>{escape(f['note'])}</td></tr>" for f in res["findings"]
    ) or '<tr><td colspan="6">Расхождений больше допуска нет.</td></tr>'
    trows = "".join(
        f"<tr><td>{escape(t['title'])}</td><td>{t['video_sessions']}</td><td>{_dur(t['video_min'])}</td>"
        f"<td>{t['paid_sessions']}</td><td>{_dur(t['paid_min'])}</td><td>{_money(t['paid_amount'], cur)}</td>"
        f"<td>{_money(t['tariff'], cur)}/ч</td></tr>" for t in res["tables"])
    unk = (f"<p class='warn'>В учёте есть столы, которых нет на камерах: {escape(', '.join(res['unknown_tables']))}"
           " — добавьте их в карту столов (--map).</p>") if res["unknown_tables"] else ""
    what = ("Сессии игры по видео (движение шаров на сукне) сравниваются с оплаченными сессиями из программы учёта."
            if res["profile"] == "billiard" else "Занятость столов по видео сравнивается с чеками из кассы.")
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Сверка с учётом</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;background:#f6f6f4;color:#1d1d1f}}
main{{max-width:960px;margin:0 auto;padding:24px 16px}} h1{{font-size:24px;margin:0 0 4px}}
.sub{{color:#666;margin-bottom:20px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.card{{background:#fff;border-radius:10px;padding:14px 16px;border-left:4px solid #c0392b}}
.card.guest{{border-left-color:#2e86c1}} .k{{font-size:13px;color:#555}} .v{{font-size:26px;font-weight:600;margin:4px 0}}
.m{{color:#333}} table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;margin-top:12px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #eee;font-size:14px}} th{{background:#fafafa;color:#555}}
h2{{font-size:18px;margin:28px 0 4px}} .note{{color:#666;font-size:13px;line-height:1.5}} .warn{{color:#a04000}}
.wrap{{overflow-x:auto}} td:nth-child(3){{white-space:nowrap}}
</style></head><body><main>
<h1>Сверка с учётом{(' — ' + escape(title)) if title else ''}</h1>
<div class="sub">Период видео: {cov_txt} · допуск по часам ±{s['tolerance_s']:.0f} с, округление {s['rounding_min']:.0f} мин,
отрезки короче {s['min_minutes']:.0f} мин не показаны</div>
<div class="cards">{''.join(cards)}</div>
<h2>Расхождения</h2><div class="wrap"><table><tr><th>Стол</th><th>Что</th><th>Когда</th><th>Сколько</th>
<th>Сумма</th><th>Пояснение</th></tr>{rows}</table></div>{unk}
<h2>По столам</h2><div class="wrap"><table><tr><th>Стол</th><th>Сессий по видео</th><th>Время по видео</th>
<th>Сессий в учёте</th><th>Оплачено, время</th><th>Оплачено, сумма</th><th>Тариф</th></tr>{trows}</table></div>
<h2>Как читать</h2><p class="note">{what} Это <b>расхождение двух источников</b>, а не вывод о том, кто виноват:
часы кассы могли уйти, сессию могли перенести на другой стол, игру могли оплатить наличными одной строкой.
Каждый отрезок можно проверить по записи регистратора за указанное время. Сумма — минуты × тариф стола.
Камеры видят движение шаров, а не людей; лица на сохранённых кадрах размыты.</p>
</main></body></html>"""


# ---------------------------------------------------------------------------------------------- CLI
def add_parser(sub) -> None:
    p = sub.add_parser("reconcile", help="сверка «занято по видео» с учётом/кассой (CSV) — B3")
    p.add_argument("--events", required=True, help="events.sqlite прогона или live")
    p.add_argument("--sessions", required=True, help="CSV оплаченных сессий / чеков (docs/billiard/RECONCILE.md)")
    p.add_argument("--map", help="YAML: столы учёта -> зоны, тарифы, допуски")
    p.add_argument("--site", help="YAML объекта: профиль, тарифы billiard.tariffs, часовой пояс")
    p.add_argument("--profile", choices=["billiard", "restaurant"], help="по умолчанию — из журнала/объекта")
    p.add_argument("--date", help="рабочий день ГГГГ-ММ-ДД, если в CSV только время (смена после полуночи — "
                                  "следующий день, до --day-start)")
    p.add_argument("--day-start", type=int, default=6, help="час начала рабочего дня (по умолчанию 6)")
    p.add_argument("--video-start", help="журнал прогона файла: местное время начала записи ГГГГ-ММ-ДД ЧЧ:ММ[:СС]")
    p.add_argument("--tolerance", type=float, help="допуск рассинхрона часов, с (по умолчанию 120)")
    p.add_argument("--rounding", type=float, help="округление в учёте, мин (по умолчанию 5)")
    p.add_argument("--min-minutes", type=float, help="не показывать отрезки короче, мин (по умолчанию 10)")
    p.add_argument("--title", default="", help="подпись в отчёте (например, дата)")
    p.add_argument("--out", help="папка для reconcile.html / .json / telegram.txt (по умолчанию рядом с журналом)")
    p.add_argument("--send", action="store_true", help="отправить строки сводки в Telegram владельцу заведения")


def cmd_reconcile(args) -> int:
    from qonaq.config import load_site
    from qonaq.events.store import EventStore

    site = load_site(args.site) if args.site else None
    mp = load_map(args.map)
    store = EventStore(args.events)
    meta = store.meta()
    profile = args.profile or (site.profile if site else None) or meta.get("profile") or "billiard"
    if profile not in ("billiard", "restaurant"):
        profile = "restaurant"
    tzname = mp.get("timezone") or (site.business.timezone if site else "Asia/Almaty")
    tz = local_tz(tzname)
    cfg = Settings(profile=profile, timezone=tzname, table_map=mp.get("tables") or {})
    if site is not None:
        cfg.tariff_per_hour = site.billiard.tariff_per_hour
        cfg.tariffs = dict(site.billiard.tariffs)
        cfg.avg_check = site.business.avg_check_kzt
        cfg.currency = site.business.currency
    for k in ("tolerance_s", "rounding_min", "min_minutes", "idle_extra_s", "tariff_per_hour", "avg_check"):
        if k in mp:
            setattr(cfg, k, float(mp[k]))
    if args.tolerance is not None:
        cfg.tolerance_s = args.tolerance
    if args.rounding is not None:
        cfg.rounding_min = args.rounding
    if args.min_minutes is not None:
        cfg.min_minutes = args.min_minutes
    offset = 0.0
    if meta.get("time_base", "video") != "epoch":
        if not args.video_start:
            print("Журнал прогона файла (время от начала видео): укажите --video-start «ГГГГ-ММ-ДД ЧЧ:ММ» — "
                  "местное время начала записи.")
            return 2
        offset = parse_dt(args.video_start, None, tz).timestamp()
    day = date.fromisoformat(args.date) if args.date else None
    paid, warn = read_paid_csv(args.sessions, day, tz, args.day_start)
    video = video_sessions(store, profile, offset)
    cov = coverage_from_store(store, offset).get("*")
    zone_type = "billiard_table" if profile == "billiard" else "table"
    known = {e["zone_id"] for e in store.events() if e["zone_id"] and e["type"] != "heartbeat"}
    if site is not None:
        from qonaq.pipeline import load_camera_zones

        for cam in site.cameras:
            known |= {z.id for z in load_camera_zones(cam).by_type(zone_type)}
    store.close()
    res = reconcile(video, paid, cfg, cov, known)
    res["warnings"] = warn
    out = Path(args.out) if args.out else Path(args.events).parent
    out.mkdir(parents=True, exist_ok=True)
    import json

    (out / "reconcile.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "reconcile.html").write_text(html_report(res, args.title, tz), encoding="utf-8")
    text = telegram_text(res, args.title, tz)
    (out / "reconcile_telegram.txt").write_text(text, encoding="utf-8")
    print(text)
    for w in warn[:10]:
        print("  ! " + w)
    print(f"-> {out / 'reconcile.html'}")
    if args.send:
        from qonaq.alerts.telegram import sender_from_env

        chat_env = site.alerts.venue_chat_env if site is not None else "TELEGRAM_VENUE_CHAT_ID"
        s = sender_from_env(chat_env)
        if s is None:
            print(f"(не отправлено: нет TELEGRAM_BOT_TOKEN / {chat_env})")
        else:
            s.send(text)
    return 0
