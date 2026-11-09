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
