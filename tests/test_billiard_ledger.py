"""Этап 3: учёт бильярдных столов по видео — рабочий день, тарифы по времени, загрузка, отчёты."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from qonaq.billiard.ledger import (
    Band,
    LedgerSettings,
    billed_minutes,
    build_ledger,
    from_store,
    open_window,
    price,
    rate_at,
    sessions_csv,
    telegram_text,
    workday_of,
    write_outputs,
)
from qonaq.events.store import EventStore
from qonaq.logic.base import Event

TZ = timezone(timedelta(hours=5))


def T(hm: str, day: int = 14) -> float:
    h, m = map(int, hm.split(":"))
    return datetime(2026, 11, day, h, m, tzinfo=TZ).timestamp()


def cfg(**kw) -> LedgerSettings:
    base = {"tariff_per_hour": 2400.0, "open_hours": "12:00-04:00", "day_start_h": 6}
    base.update(kw)
    return LedgerSettings(**base)


# ---------------------------------------------------------------- рабочий день
def test_night_game_belongs_to_previous_workday():
    assert workday_of(T("01:10", 15), TZ) == date(2026, 11, 14)
    assert workday_of(T("06:00", 15), TZ) == date(2026, 11, 15)
    a, b = open_window(date(2026, 11, 14), "12:00-04:00", TZ)
    assert (a, b) == (T("12:00"), T("04:00", 15))


def test_game_over_midnight_counted_whole_in_its_start_day():
    days = build_ledger({"B2": [(T("23:30"), T("01:10", 15))]}, cfg(), TZ)
    assert [d["day"] for d in days] == ["2026-11-14"]
    s = days[0]["sessions"][0]
    assert s["minutes"] == 100 and s["amount"] == 4000  # 100 мин × 2400 ₸/ч


# ---------------------------------------------------------------- тарифы
def test_tariff_band_splits_session_by_minute():
    c = cfg(bands=[Band(18 * 60, 2 * 60, 3600.0)])  # вечер 18:00–02:00 — 3600 ₸/ч
    amount, by_rate = price("B1", T("17:30"), T("18:30"), c, TZ)
    assert by_rate == {2400.0: 30.0, 3600.0: 30.0}
    assert round(amount) == 1200 + 1800


def test_band_over_midnight_uses_weekday_of_band_start():
    # пятница 13.11.2026 — 5; ночь на субботу до 02:00 — всё ещё пятничная полоса
    c = cfg(bands=[Band(20 * 60, 2 * 60, 5000.0, weekdays=(5,))])
    assert rate_at("B1", datetime(2026, 11, 14, 1, 0, tzinfo=TZ), c) == 5000.0  # суббота 01:00 → пятничная ночь
    assert rate_at("B1", datetime(2026, 11, 14, 21, 0, tzinfo=TZ), c) == 2400.0  # суббота вечер — не пятница


def test_band_only_for_listed_tables_and_table_tariff_fallback():
    c = cfg(tariffs={"B5": 6000.0}, bands=[Band(0, 1439, 1000.0, tables=("B1",))])
    t = datetime(2026, 11, 14, 15, 0, tzinfo=TZ)
    assert rate_at("B1", t, c) == 1000.0 and rate_at("B5", t, c) == 6000.0 and rate_at("B2", t, c) == 2400.0


def test_club_rounding_and_minimum():
    c = cfg(rounding_min=10, min_minutes=30)
    assert billed_minutes(12, c) == 30 and billed_minutes(41, c) == 50 and billed_minutes(0, c) == 0
    d = build_ledger({"B1": [(T("20:00"), T("20:41"))]}, c, TZ)[0]
    assert d["sessions"][0]["billed_minutes"] == 50 and d["sessions"][0]["amount"] == 2000


# ---------------------------------------------------------------- загрузка, простой, слепые минуты
def test_utilization_excludes_minutes_system_did_not_see():
    open_s = T("04:00", 15) - T("12:00")  # 16 ч
    cov = [(T("12:00"), T("16:00")), (T("18:00"), T("04:00", 15))]  # 2 ч камера не отвечала
    d = build_ledger({"B1": [(T("20:00"), T("22:00"))]}, cfg(), TZ, coverage=cov, tables={"B1", "B2"})[0]
    t1, t2 = d["tables"]
    assert t1["zone"] == "B1" and t1["util"] == round(2 / 14, 3) and t1["idle_open_min"] == 12 * 60
    assert t2["games"] == 0 and t2["util"] == 0.0 and t2["idle_open_min"] == 14 * 60  # стол без игр — тоже в отчёте
    assert d["totals"]["blind_open_min"] == 120 and open_s == 16 * 3600
    assert any("не видела" in n for n in d["notes"])


def test_play_outside_open_hours_is_flagged():
    d = build_ledger({"B1": [(T("03:30", 15), T("05:00", 15))]}, cfg(), TZ)[0]
    assert d["day"] == "2026-11-14" and d["sessions"][0]["outside_open_min"] == 60
    assert any("вне часов работы" in n for n in d["notes"])


def test_hourly_grid_and_peak():
    d = build_ledger({"B1": [(T("20:30"), T("22:00"))], "B2": [(T("21:00"), T("21:45"))]}, cfg(), TZ)[0]
    hs = d["hours"]
    i = hs["labels"].index("21:00")
    assert hs["by_table"]["B1"][i] == 60 and hs["by_table"]["B2"][i] == 45 and hs["by_table"]["B1"][i - 1] == 30
    assert d["peak"]["label"] == "21:00–22:00"


# ---------------------------------------------------------------- из журнала и отчёты
def _store(tmp_path, events, heartbeat=None) -> EventStore:
    st = EventStore(tmp_path / "events.sqlite")
    st.set_meta(time_base="epoch", profile="billiard")
    st.add(events)
    if heartbeat:
        a, b = heartbeat
        st.add([Event(t, "", "heartbeat", {"window_s": 60, "up": ["hall_1"]}, "hall_1")
                for t in range(int(a) + 60, int(b) + 1, 60)])
    return st


def test_from_store_open_game_counted_to_last_heartbeat(tmp_path):
    ev = [Event(T("20:00"), "B1", "game_started", {"start": T("20:00"), "detected_ts": T("20:01")}, "hall_1"),
          Event(T("21:30"), "B1", "game_ended", {"start": T("20:00"), "end": T("21:30")}, "hall_1"),
          Event(T("22:00"), "B2", "game_started", {"start": T("22:00"), "detected_ts": T("22:02")}, "hall_1")]
    st = _store(tmp_path, ev, heartbeat=(T("12:00"), T("23:00")))
    days = from_store(st, cfg(), TZ, tables={"B1", "B2", "B3"})
    st.close()
    d = days[0]
    assert d["totals"]["games"] == 2 and d["totals"]["play_min"] == 90 + 60
    b2 = next(s for s in d["sessions"] if s["zone"] == "B2")
    assert b2["still_playing"] and any("ещё идёт" in n for n in d["notes"])
    txt = telegram_text(d, "Клуб")
    assert "Бильярд · 14 ноября · Клуб" in txt and "Стол 3 — не играли" in txt and "≈ 6 000 ₸" in txt
    paths = write_outputs(days, tmp_path / "ledger", TZ, "тест")
    assert all(p.exists() for p in paths.values())
    assert "Стол 2" in paths["html"].read_text(encoding="utf-8")
    csv_text = sessions_csv(days, TZ)
    assert csv_text.startswith("﻿Рабочий день;Стол;Начало") and "2026-11-14 20:00;2026-11-14 21:30;90" in csv_text


def test_cli_ledger_on_file_run_needs_video_start(tmp_path, capsys):
    from qonaq.cli import main

    st = EventStore(tmp_path / "events.sqlite")
    st.set_meta(time_base="video", profile="billiard", scene_s=3600)
    st.add([Event(600, "B1", "game_started", {"start": 600.0}), Event(2400, "B1", "game_ended",
                                                                       {"start": 600.0, "end": 2400.0})])
    st.close()
    args = ["billiard", "ledger", "--events", str(tmp_path / "events.sqlite"), "--out", str(tmp_path / "o")]
    assert main(args) == 2
    assert main(args + ["--video-start", "2026-11-14 19:00", "--timezone", "Asia/Almaty"]) == 0
    out = capsys.readouterr().out
    assert "Игра: 30 мин" in out and (tmp_path / "o" / "ledger.html").exists()
