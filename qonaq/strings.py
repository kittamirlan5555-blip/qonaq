"""Все пользовательские строки — здесь (русский; задел под казахский: добавить словарь KK)."""
from __future__ import annotations

RU = {
    "state.FREE": "Свободен",
    "state.SEATED_WAITING": "Ждут официанта",
    "state.SERVED": "Обслуживаются",
    "state.NEEDS_CLEANING": "Не убран",
    "short.FREE": "СВОБОДЕН",
    "short.SEATED_WAITING": "ЖДУТ",
    "short.SERVED": "ОБСЛУЖЕН",
    "short.NEEDS_CLEANING": "НЕ УБРАН",
    "role.guest": "гость",
    "role.staff": "персонал",
    "role.unknown": "?",
    "alert.no_waiter": "{zone}: гости ждут официанта {mmss}",
    "alert.dirty_table": "{zone}: стол не убран {mmss}",
    "alert.long_service_without_visit": "{zone}: к гостям давно не подходили ({mmss})",
    "alert.occupied_without_check": "{zone}: стол занят {mmss}, а чек в POS не открыт",
    "alert.desk_unattended": "{zone}: у стойки очередь, а сотрудника нет {mmss}",
    "alert.long_wait": "{zone}: посетитель ждёт уже {mmss}",
    # S4: тексты Telegram — коротко, номер стола и время, без технических слов
    "tg.no_waiter": "{zone}: гости ждут официанта уже {dur} (сели в {since})",
    "tg.dirty_table": "{zone} не убран уже {dur} (гости ушли в {since})",
    "tg.long_service_without_visit": "{zone}: к гостям не подходили {dur}",
    "tg.occupied_without_check": "{zone} занят {dur}, а счёт в кассе не открыт",
    "tg.desk_unattended": "{zone}: очередь, а сотрудника на месте нет уже {dur}",
    "tg.long_wait": "{zone}: посетитель ждёт уже {dur} (с {since})",
    "tg.late": "Доставлено с опозданием: не было интернета.",
    "panel.title": "Qonaq Vision",
    "panel.load": "Загрузка зала",
    "panel.tables": "Столы",
    "panel.time": "Время",
    "panel.waiting": "ждут",
    "panel.dirty": "не убрано",
    "panel.free": "свободно",
    "panel.occupied": "занято",
    "panel.people": "Людей в кадре",
    "panel.queue": "Очередь",
    "panel.wait_now": "Ожидают",
    "panel.desk": "Стойка",
    "zone.entrance": "Вход",
    "zone.waiting_area": "Ожидание",
    "zone.service_point": "Стойка",
}

_LANG = RU


def t(key: str, **kw) -> str:
    s = _LANG.get(key, key)
    return s.format(**kw) if kw else s


def guests_word(n: int) -> str:
    """1 гость, 2 гостя, 5 гостей."""
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return f"{n} гость"
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return f"{n} гостя"
    return f"{n} гостей"


def fmt_money(v: float, currency: str = "₸") -> str:
    """45000 -> '45 000 ₸' (неразрывные пробелы не используем — Telegram/markdown дружелюбно)."""
    return f"{int(round(v)):,}".replace(",", " ") + f" {currency}"


def alert_text(kind: str, zone: str, seconds: float) -> str:
    from qonaq.logic.base import fmt_mmss

    return t(f"alert.{kind}", zone=zone, mmss=fmt_mmss(seconds))


def human_dur(seconds: float) -> str:
    """Для сообщений людям: «40 с», «3 мин», «1 ч 05 мин»."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s} с"
    m = s // 60
    if m < 60:
        return f"{m} мин"
    return f"{m // 60} ч {m % 60:02d} мин"


def table_label(title: str) -> str:
    """'T05' -> 'Стол 5', 'Стол у окна' -> как есть. Зоны обычно размечены T01..T20."""
    import re

    m = re.fullmatch(r"[A-Za-zА-Яа-я]{0,2}[-_ ]?0*(\d+)", title.strip())
    return f"Стол {m.group(1)}" if m else title


def clock(wall: float, tz: str | None = None) -> str:
    """Местное время ЧЧ:ММ (часовой пояс объекта; на Windows без tzdata — время компьютера)."""
    import time
    from datetime import datetime

    try:
        from zoneinfo import ZoneInfo

        return datetime.fromtimestamp(wall, ZoneInfo(tz)).strftime("%H:%M") if tz else time.strftime(
            "%H:%M", time.localtime(wall))
    except Exception:  # noqa: BLE001
        return time.strftime("%H:%M", time.localtime(wall))


def alert_message(kind: str, zone_title: str, seconds: float, wall: float | None = None,
                  tz: str | None = None) -> str:
    """Текст алерта для администратора в Telegram: «⚠ 14:35 · Стол 5: гости ждут официанта уже 3 мин (сели в 14:32)»."""
    zone = table_label(zone_title)
    since = clock(wall - seconds, tz) if wall is not None else ""
    key = f"tg.{kind}"
    if key not in _LANG:
        return f"⚠ {alert_text(kind, zone, seconds)}"
    body = t(key, zone=zone, dur=human_dur(seconds), since=since)
    if wall is None:
        body = body.split(" (сели в")[0].split(" (гости ушли в")[0].split(" (с ")[0]
        return f"⚠ {body}"
    return f"⚠ {clock(wall, tz)} · {body}"


def plural(n: int, one: str, few: str, many: str) -> str:
    """plural(3, 'стол', 'стола', 'столов') -> '3 стола'."""
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return f"{n} {one}"
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"
