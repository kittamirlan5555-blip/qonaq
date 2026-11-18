"""Сообщения в Telegram как часть продукта: строгие HTML-карточки алертов, кнопки, правка исходного алерта, итог смены.

Один источник для live (`AlertDispatcher` с отправителем, умеющим HTML) и для показа на записи
(`scripts/demo_telegram_shift.py`): `TelegramFeed.handle(events)` превращает события системы в действия
«отправить» / «отредактировать». Тексты руками не пишутся — только из событий.

Алерт:
    <b>Стол 4 · гости ждут официанта 3 мин</b>
    Сели в 19:42, к столу пока никто не подходил.
    [Принял] [Неверно]
Когда официант подошёл (событие served той же сессии стола) — editMessageText, строка ниже:
    <i>Подошли через 3 мин 40 с</i>
«Стол не убран» закрывается событием cleaned: <i>Убрали через 9 мин</i>.
«К гостям давно не подходили» — следующим визитом персонала: <i>Подошли через 2 мин</i>.

Итог смены (`shift_report` → `shift_html`) считается из сессий и событий журнала; строка, которую нельзя честно
посчитать (нет ни одного измерения), не выводится.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from html import escape
from typing import Any

from qonaq.strings import table_label

Clock = Callable[[float], str]  # время журнала -> «ЧЧ:ММ» (время сцены: live — местное, запись — по OSD)

# Какое событие закрывает алерт и какой строкой
RESOLVES = {"no_waiter": "served", "dirty_table": "cleaned", "long_service_without_visit": "staff_visit"}


def dur(seconds: float) -> str:
    """40 с · 3 мин 40 с · 12 мин · 1 ч 05 мин (до 10 минут — с секундами, если они есть)."""
    s = max(0, int(seconds + 0.5))
    if s < 60:
        return f"{s} с"
    m, sec = divmod(s, 60)
    if m < 10 and sec:
        return f"{m} мин {sec} с"
    if m < 60:
        return f"{m} мин"
    return f"{m // 60} ч {m % 60:02d} мин"


def mins(seconds: float) -> str:
    """Порог алерта в заголовке: целые минуты («3 мин»), час и больше — «1 ч 05 мин»."""
    m = max(1, int(seconds // 60))
    return f"{m} мин" if m < 60 else f"{m // 60} ч {m % 60:02d} мин"


def alert_card(kind: str, table: str, seconds: float, since: str | None, icon: str = "") -> str:
    """HTML-карточка алерта: жирный заголовок «Стол N · что случилось», под ним — короткий контекст."""
    t = escape(table)
    pre = f"{icon} " if icon else ""
    if kind == "no_waiter":
        head, ctx = f"{t} · гости ждут официанта {mins(seconds)}", (
            f"Сели в {since}, к столу пока никто не подходил." if since else "К столу пока никто не подходил.")
    elif kind == "dirty_table":
        head = f"{t} · не убран {mins(seconds)}"
        ctx = (f"Гости ушли в {since}, после этого к столу не подходили." if since
               else "После гостей к столу не подходили.")
    elif kind == "long_service_without_visit":
        head, ctx = f"{t} · к гостям не подходили {mins(seconds)}", (
            f"Последний подход — в {since}." if since else "")
    elif kind == "occupied_without_check":
        head, ctx = f"{t} · занят {mins(seconds)}, счёт не открыт", ""
    else:
        head, ctx = f"{t} · {escape(kind)} {mins(seconds)}", ""
    return f"{pre}<b>{head}</b>" + (f"\n{ctx}" if ctx else "")


def resolution_line(kind: str, seconds: float) -> str:
    if kind == "dirty_table":
        return f"<i>Убрали через {dur(seconds)}</i>"
    return f"<i>Подошли через {dur(seconds)}</i>"


def keyboard(alert_id: str, acked: bool = False) -> dict[str, Any]:
    row = [] if acked else [{"text": "Принял", "callback_data": f"ack:{alert_id}"}]
    row.append({"text": "Неверно", "callback_data": f"fb:{alert_id}"})
    return {"inline_keyboard": [row]}


@dataclass
class Action:
    op: str  # send | edit
    key: tuple  # (камера, стол, вид алерта, сессия) — какое сообщение правим
    html: str
    markup: dict[str, Any] | None
    ts: float  # время журнала, когда это произошло
    event: dict[str, Any] = field(default_factory=dict)


def _get(e, name, default=None):
    return e.get(name, default) if isinstance(e, dict) else getattr(e, name, default)


class TelegramFeed:
    """События системы → сообщения. Помнит открытые алерты, чтобы дописать в них подтверждение."""

    def __init__(self, clock: Clock, titles: dict[str, str] | None = None, icon: str = "",
                 id_of: Callable[[dict], str] | None = None):
        self.clock = clock
        self.titles = titles or {}
        self.icon = icon
        self.id_of = id_of or (lambda e: f"{e['camera_id']}:{e['zone_id']}:{e['kind']}:{e['sid']}")
        self.open: dict[tuple, dict[str, Any]] = {}  # key -> {html, alert_id, kind, ts, acked}
        self._seated: dict[tuple, float] = {}  # (камера, стол, сессия) -> время посадки
        self._left: dict[tuple, float] = {}
        self._visit: dict[tuple, float] = {}

    def title(self, zone: str) -> str:
        return self.titles.get(zone) or table_label(zone)

    def handle(self, events: Iterable[Any]) -> list[Action]:
        out: list[Action] = []
        for e in events:
            typ, zone = _get(e, "type"), _get(e, "zone_id")
            cam, ts = _get(e, "camera_id", "") or "", float(_get(e, "ts"))
            p = _get(e, "payload") or {}
            sid = p.get("sid")
            sk = (cam, zone, sid)
            if typ == "seated":
                self._seated[sk] = ts
            elif typ == "left":
                self._left[sk] = ts
            elif typ in ("served", "staff_visit"):
                self._visit[sk] = ts
            if typ == "alert":
                kind = p.get("kind", "")
                key = (cam, zone, kind, sid, p.get("seq"))
                if key in self.open:
                    continue
                since = {"no_waiter": self._seated, "dirty_table": self._left,
                         "long_service_without_visit": self._visit}.get(kind, {}).get(sk)
                seconds = float(next((p[k] for k in ("waited_s", "dirty_s", "since_visit_s", "occupied_s")
                                      if k in p), 0.0))
                if since is None:
                    since = ts - seconds if seconds else None
                html = alert_card(kind, self.title(zone), seconds, self.clock(since) if since is not None else None,
                                  self.icon)
                aid = self.id_of({"camera_id": cam, "zone_id": zone, "kind": kind, "sid": sid})
                self.open[key] = {"html": html, "alert_id": aid, "kind": kind, "ts": ts, "acked": False,
                                  "since": since}
                out.append(Action("send", key, html, keyboard(aid), ts, {"zone": zone, "kind": kind}))
                continue
            for key, a in list(self.open.items()):
                if key[0] != cam or key[1] != zone or key[3] != sid or RESOLVES.get(a["kind"]) != typ:
                    continue
                if a["kind"] == "no_waiter":
                    secs = float(p.get("response_time_s") or (ts - (a["since"] or ts)))
                elif a["kind"] == "dirty_table":
                    secs = float(p.get("cleanup_time_s") or (ts - (a["since"] or ts)))
                else:
                    secs = ts - a["ts"]
                a["html"] = a["html"] + "\n" + resolution_line(a["kind"], secs)
                out.append(Action("edit", key, a["html"], keyboard(a["alert_id"], acked=True), ts,
                                  {"zone": zone, "kind": a["kind"], "resolved_s": secs}))
                del self.open[key]
        return out


# ------------------------------------------------------------------ итог смены
def shift_report(sessions: list[dict[str, Any]], events: list[dict[str, Any]], clock: Clock,
                 hour_of: Callable[[float], int], titles: dict[str, str] | None = None,
                 wait_threshold_s: float = 180.0, dirty_threshold_s: float = 300.0,
                 t_end: float | None = None) -> dict[str, Any]:
    """Цифры итога смены — только из сессий и событий журнала. Сессии, начатые до записи (preexisting), в посадки и
    время за столом не входят: их начало система не видела. `t_end` у сессии — до какого момента система за этим
    столом смотрела (показ из отрезков); дальше о столе ничего не утверждаем."""
    titles = titles or {}
    name = lambda z: titles.get(z) or table_label(z)  # noqa: E731
    new = [s for s in sessions if not s.get("preexisting")]
    stay = [s["visit_duration_s"] for s in new if s.get("visit_duration_s") and s.get("left_ts")]
    resp = [s for s in new if s.get("response_time_s") is not None]
    worst = max(resp, key=lambda s: s["response_time_s"], default=None)
    end = t_end if t_end is not None else max([e["ts"] for e in events] + [0.0])
    long_wait = sum(1 for s in resp if s["response_time_s"] > wait_threshold_s)
    long_wait += sum(1 for s in new if s.get("served_ts") is None and s.get("left_ts") is None
                     and s.get("t_end", end) - s["seated_ts"] > wait_threshold_s)
    cleaned = [s for s in sessions if s.get("cleanup_time_s") is not None]
    dirty_open = [s.get("t_end", end) - s["left_ts"] for s in sessions if s.get("left_ts")
                  and s.get("cleaned_ts") is None and s.get("t_end", end) > s["left_ts"]]
    dirty_all = [s["cleanup_time_s"] for s in cleaned] + dirty_open
    alerts = [e for e in events if e["type"] == "alert"]
    by_table: dict[str, int] = {}
    for a in alerts:
        by_table[a["zone_id"]] = by_table.get(a["zone_id"], 0) + 1
    busy: dict[int, float] = {}
    for s in sessions:  # стол-минуты занятости по часам сцены
        a, b = float(s["seated_ts"]), float(s.get("left_ts") or s.get("t_end", end))
        t = a
        while t < b:
            h = hour_of(t)
            nxt = min(b, t + 60.0)
            busy[h] = busy.get(h, 0.0) + (nxt - t)
            t = nxt
    peak = max(busy, key=busy.get) if busy else None
    return {
        "seatings": len(new),
        "stay_mean_s": sum(stay) / len(stay) if stay else None,
        "resp_n": len(resp),
        "resp_mean_s": sum(s["response_time_s"] for s in resp) / len(resp) if resp else None,
        "resp_worst": None if worst is None else {"s": worst["response_time_s"], "table": name(worst["zone_id"]),
                                                  "at": clock(worst["seated_ts"])},
        "long_wait_n": long_wait,
        "clean_mean_s": sum(s["cleanup_time_s"] for s in cleaned) / len(cleaned) if cleaned else None,
        "dirty_over_n": sum(1 for x in dirty_all if x > dirty_threshold_s),
        "dirty_total_s": sum(dirty_all) if dirty_all else None,
        "peak_hour": peak,
        # «чаще всего» — только если у стола хотя бы два алерта, иначе это не закономерность
        "problem_table": (name(max(by_table, key=by_table.get))
                          if by_table and max(by_table.values()) >= 2 else None),
        "alerts": len(alerts),
    }


def shift_html(r: dict[str, Any], site: str, day: str) -> str:
    """Итог смены строгим HTML: разделы с жирными заголовками; строки без данных не выводятся."""
    from qonaq.strings import plural

    def times(n: int) -> str:
        return plural(n, "раз", "раза", "раз")

    sec: list[tuple[str, list[str]]] = []
    g = []
    if r["seatings"]:
        g.append(f"Посадок за столы — {r['seatings']}")
    if r["stay_mean_s"]:
        g.append(f"Среднее время за столом — {dur(r['stay_mean_s'])}")
    sec.append(("Гости", g))
    s = []
    if r["resp_mean_s"] is not None:
        s.append(f"Подход официанта в среднем — {dur(r['resp_mean_s'])}")
    if r["resp_worst"] and r["resp_n"] > 1:
        w = r["resp_worst"]
        s.append(f"Дольше всего ждали — {dur(w['s'])}, {w['table'].lower()} в {w['at']}")
    if r["resp_n"] and r["long_wait_n"]:
        s.append(f"Ждали дольше 3 мин — {times(r['long_wait_n'])}")
    sec.append(("Обслуживание", s))
    t = []
    if r["clean_mean_s"] is not None:
        t.append(f"Уборка после гостей в среднем — {dur(r['clean_mean_s'])}")
    if r["dirty_over_n"]:
        t.append(f"Стояли неубранными дольше 5 мин — {times(r['dirty_over_n'])}")
    if r["dirty_total_s"]:
        t.append(f"Простой неубранных столов за смену — {dur(r['dirty_total_s'])}")
    sec.append(("Столы", t))
    w2 = []
    if r["peak_hour"] is not None:
        h = r["peak_hour"]
        w2.append(f"Самый загруженный час — {h:02d}:00–{(h + 1) % 24:02d}:00")
    if r["problem_table"]:
        w2.append(f"Чаще всего проблемы — {r['problem_table'].lower()}")
    sec.append(("Где проседает", w2))
    out = ["<b>Итоги смены</b>", f"{escape(site)} · {escape(day)}"]
    for head, lines in sec:
        if lines:
            out += ["", f"<b>{head}</b>"] + [escape(x) for x in lines]
    return "\n".join(out)
