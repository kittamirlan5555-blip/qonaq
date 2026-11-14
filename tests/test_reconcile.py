"""B3: сверка «занято по видео» с учётом (CSV) — бильярд и ресторан."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from qonaq.events.store import EventStore
from qonaq.logic.base import Event
from qonaq.reconcile import (
    Paid,
    Settings,
    coverage_from_store,
    html_report,
    read_paid_csv,
    reconcile,
    telegram_text,
)

TZ = timezone(timedelta(hours=5))


def T(hm: str, day: int = 14) -> float:
    h, m = map(int, hm.split(":"))
    d = datetime(2026, 11, day, h, m, tzinfo=TZ)
    return d.timestamp()


def kinds(res):
    return [(f["kind"], f["zone"], round(f["minutes"])) for f in res["findings"]]


def test_csv_semicolon_cp1251_time_only_over_midnight(tmp_path):
    p = tmp_path / "paid.csv"
    p.write_bytes("Стол;Начало;Конец;Сумма, тг\nСтол 4;21:10;22:40;3 750,00\nСтол 2;23:30;01:10;4166\n"
                  "Стол 1;00:20;01:00;1000\n".encode("cp1251"))
    rows, warn = read_paid_csv(p, date(2026, 11, 14), TZ)
    assert not warn and len(rows) == 3
    assert rows[0].table == "Стол 4" and rows[0].start == T("21:10") and rows[0].amount == 3750.0
    assert rows[1].end == T("01:10", 15)  # конец после полуночи — следующий день
    assert rows[2].start == T("00:20", 15)  # смена после полуночи (до 06:00) — следующий календарный день


def test_csv_comma_full_datetime_and_duration(tmp_path):
    p = tmp_path / "paid.csv"
    p.write_text("table,start,duration,amount\nB1,14.11.2026 20:00,1:30,3750\nB2,2026-11-14 21:00,45,\n",
                 encoding="utf-8-sig")
    rows, _ = read_paid_csv(p, None, TZ)
    assert rows[0].end - rows[0].start == 5400 and rows[1].end - rows[1].start == 2700 and rows[1].amount is None


def test_csv_time_without_date_needs_date(tmp_path):
    p = tmp_path / "paid.csv"
    p.write_text("стол,начало,конец\n1,21:00,22:00\n", encoding="utf-8")
    rows, warn = read_paid_csv(p, None, TZ)
    assert rows == [] and "--date" in warn[0]


def test_csv_missing_columns(tmp_path):
    p = tmp_path / "paid.csv"
    p.write_text("что,когда\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_paid_csv(p, None, TZ)


def _cfg(**kw):
    return Settings(profile=kw.pop("profile", "billiard"), tariff_per_hour=2400, **kw)


def test_billiard_three_kinds_and_money():
    video = {"B4": [(T("21:10"), T("22:40"))],  # оплаты нет вообще
             "B2": [(T("20:00"), T("21:30"))],  # оплачено 20:00–21:00 → 30 мин сверху
             "B1": [(T("19:00"), T("19:40"))]}  # оплачено 19:00–20:30 → стол пустой 50 мин
    paid = [Paid("Стол 2", T("20:00"), T("21:00")), Paid("Стол 1", T("19:00"), T("20:30"))]
    res = reconcile(video, paid, _cfg())
    got = kinds(res)
    assert ("unpaid_play", "B4", 90) in got
    assert any(k == "overtime" and z == "B2" and 20 <= m <= 30 for k, z, m in got)
    assert any(k == "idle_paid" and z == "B1" and 40 <= m <= 50 for k, z, m in got)
    unpaid = [f for f in res["findings"] if f["kind"] == "unpaid_play"][0]
    assert unpaid["money"] == pytest.approx(90 * 40)  # 2400 ₸/ч = 40 ₸/мин
    text = telegram_text(res, "14.11", TZ)
    assert "Стол 4, 21:10–22:40 (1 ч 30 мин) — по учёту не оплачено" in text
    for bad in ("админ", "администратор", "украл", "виноват"):
        assert bad not in text.lower().replace("не вывод о сотрудниках", "")


def test_tolerance_and_rounding_absorb_small_differences():
    video = {"B1": [(T("20:00") - 90, T("21:00") + 200)]}  # часы кассы на 1,5 мин позади; учёт округлил вверх
    paid = [Paid("1", T("20:00"), T("21:00"))]
    assert reconcile(video, paid, _cfg(min_minutes=1))["findings"] == []
    res = reconcile(video, paid, _cfg(tolerance_s=0, rounding_min=0, min_minutes=1))
    assert {f["kind"] for f in res["findings"]} == {"overtime"}


def test_short_segments_hidden():
    video = {"B1": [(T("20:00"), T("20:06"))]}
    assert reconcile(video, [], _cfg(min_minutes=10))["findings"] == []


def test_blind_period_is_not_idle_table():
    """Система не работала 20:30–21:30 (перезапуск/свет): «счётчик шёл, стол пустой» там не утверждаем."""
    video = {"B1": [(T("20:00"), T("20:30"))]}
    paid = [Paid("1", T("20:00"), T("21:30"))]
    cov = [(T("19:00"), T("20:30")), (T("21:30"), T("23:00"))]
    assert reconcile(video, paid, _cfg(), cov)["findings"] == []
    assert [f["kind"] for f in reconcile(video, paid, _cfg())["findings"]] == ["idle_paid"]


def test_unknown_table_and_outside_coverage():
    paid = [Paid("VIP", T("20:00"), T("21:00")), Paid("Стол 1", T("10:00"), T("11:00"))]
    res = reconcile({"B1": []}, paid, _cfg(), [(T("19:00"), T("23:00"))])
    assert res["unknown_tables"] == ["VIP"] and res["outside_coverage"] == 1 and res["findings"] == []


def test_table_map_and_tariff():
    cfg = _cfg(table_map={"Русский 1": {"zone": "B7", "tariff": 6000, "name": "Русский бильярд 1"}})
    res = reconcile({"B7": [(T("20:00"), T("21:00"))]}, [], cfg)
    f = res["findings"][0]
    assert f["table"] == "Русский бильярд 1" and f["money"] == pytest.approx(6000)


def test_restaurant_table_busy_without_check():
    video = {"T03": [(T("13:00"), T("13:50"))], "T05": [(T("13:00"), T("14:00"))]}
    paid = [Paid("Стол 5", T("13:10"), T("13:40"), 18000)]  # чек открыт позже посадки, закрыт до ухода — норма
    res = reconcile(video, paid, Settings(profile="restaurant", avg_check=15000))
    assert kinds(res) == [("no_check", "T03", 50)]
    html = html_report(res, "тест", TZ)
    assert "Стол занят, чека нет" in html and "Играли дольше" not in html


def test_coverage_from_heartbeats_and_cli(tmp_path, capsys):
    from qonaq.cli import main

    db = tmp_path / "events.sqlite"
    st = EventStore(db)
    st.set_meta(time_base="epoch", profile="billiard")
    hb = [Event(T("19:00") + 60 * k, "", "heartbeat", {"window_s": 60, "up": ["hall_1"]}) for k in range(1, 240)
          if not (90 <= k < 150)]  # 20:30–21:30 система не работала
    st.add(hb)
    st.add([Event(T("19:05"), "B1", "lamp_on", {}, "hall_1"),
            Event(T("21:40"), "B4", "game_started", {"start": T("21:40")}, "hall_1"),
            Event(T("22:40"), "B4", "game_ended", {"start": T("21:40"), "end": T("22:40")}, "hall_1")])
    cov = coverage_from_store(st)["*"]
    assert len(cov) == 2
    st.close()
    csv = tmp_path / "paid.csv"
    csv.write_text("стол;начало;конец;сумма\nСтол 1;20:00;21:20;3000\n", encoding="utf-8")
    rc = main(["reconcile", "--events", str(db), "--sessions", str(csv), "--date", "2026-11-14", "--title", "14.11",
               "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert rc == 0 and "Стол 4, 21:40–22:40" in out and "Счётчик шёл" in out
    assert (tmp_path / "out" / "reconcile.html").exists()
