"""Машина состояний стола на синтетических таймлайнах (TASKS M3)."""
import pytest

from qonaq.config import AlertThresholds, RestaurantThresholds
from qonaq.logic.base import PresenceFilter
from qonaq.logic.restaurant import TableMachine, TableState
from tests.helpers import Timeline

G = (1, 2)  # два гостя
W = (50,)  # официант


def machine(**th) -> TableMachine:
    # В тестах «окно старта» короткое (5 с), чтобы сценарии не начинались с 30 секунд пустоты;
    # значение по умолчанию (30 с) проверяется отдельно в test_preexisting_grace_default.
    th.setdefault("preexisting_grace_s", 5.0)
    return TableMachine("T01", RestaurantThresholds(**th), AlertThresholds())


def seated(tl: Timeline, pre_free: float = 20.0) -> Timeline:
    """Пустой стол pre_free секунд, затем гости сели и сидят 20 с."""
    return tl.run(pre_free).run(20, guests=G)


# ---------------------------------------------------------------- PresenceFilter
def test_presence_filter_needs_sustained_presence():
    f = PresenceFilter(5.0, 0.7, 0.2)
    for i in range(10):  # 2 с присутствия — мало
        f.update(i * 0.2, True)
    assert not f.state
    for i in range(10, 30):
        f.update(i * 0.2, True)
    assert f.state and f.since == pytest.approx(0.0)


def test_presence_filter_backdates_off():
    f = PresenceFilter(5.0, 0.7, 0.2)
    t = 0.0
    for _ in range(50):
        f.update(t, True)
        t += 0.2
    last_true = t - 0.2
    for _ in range(50):
        f.update(t, False)
        t += 0.2
    assert not f.state and f.since == pytest.approx(last_true)


def test_presence_filter_survives_flicker():
    f = PresenceFilter(5.0, 0.7, 0.2)
    for i in range(200):
        f.update(i * 0.2, i % 4 != 0)  # 25% пропусков
    assert f.state


# ---------------------------------------------------------------- базовый сценарий
def test_full_visit_cycle():
    tl = seated(Timeline(machine()))
    assert tl.m.state == TableState.SEATED_WAITING
    tl.run(40, guests=G)  # ждут
    tl.run(8, guests=G, staff=W)  # подошёл официант
    assert tl.m.state == TableState.SERVED
    served = tl.of("served")[0]
    assert served.payload["response_time_s"] == pytest.approx(60.0, abs=1.0)
    tl.run(120, guests=G)  # едят
    tl.run(6, guests=G, staff=W)  # ещё визит
    tl.run(60, guests=G)
    tl.run(100)  # ушли
    assert tl.m.state == TableState.NEEDS_CLEANING
    left = tl.of("left")[0]
    assert left.ts == pytest.approx(20 + 20 + 40 + 8 + 120 + 6 + 60, abs=1.0)  # задним числом
    tl.run(30)
    tl.run(15, staff=W)  # уборка
    assert tl.m.state == TableState.FREE
    cleaned = tl.of("cleaned")[0]
    assert cleaned.payload["cleanup_time_s"] == pytest.approx(100 + 30 + 10, abs=1.5)
    assert tl.types() == ["seated", "served", "staff_visit", "left", "cleaned"]


def test_seated_is_backdated_to_arrival():
    tl = seated(Timeline(machine()))
    ev = tl.of("seated")[0]
    assert ev.ts == pytest.approx(20.0, abs=0.3)
    assert ev.payload["detected_ts"] >= 35.0 - 0.3
    assert ev.payload["preexisting"] is False


def test_state_changes_are_logged():
    tl = seated(Timeline(machine())).run(10, guests=G, staff=W)
    sc = [(e.payload["from"], e.payload["to"]) for e in tl.of("state_change")]
    assert sc == [("FREE", "SEATED_WAITING"), ("SEATED_WAITING", "SERVED")]


# ---------------------------------------------------------------- шум
def test_guest_smoke_break_60s_keeps_table():
    tl = seated(Timeline(machine())).run(8, guests=G, staff=W).run(60, guests=G)
    tl.run(60)  # вышли покурить на 60 с (< leave_min_s=90)
    assert tl.m.state == TableState.SERVED
    tl.run(60, guests=G)
    assert tl.m.state == TableState.SERVED
    assert "left" not in tl.types()


def test_flickering_detection_does_not_break_session():
    tl = Timeline(machine()).run(20)
    tl.run(60, guests=G, flicker=4)  # каждое четвёртое наблюдение пустое (75% >= порога 70%)
    assert tl.m.state == TableState.SEATED_WAITING
    assert len(tl.of("seated")) == 1
    tl.run(10, guests=G, staff=W, flicker=4)
    assert tl.m.state == TableState.SERVED
    tl.run(120, guests=G, flicker=4)
    assert tl.types() == ["seated", "served"]


def test_short_total_dropout_is_tolerated():
    tl = seated(Timeline(machine())).run(3).run(30, guests=G)  # 3 с полной потери детекции
    assert tl.m.state == TableState.SEATED_WAITING
    assert len(tl.of("seated")) == 1


def test_passerby_through_free_table_is_ignored():
    tl = Timeline(machine()).run(20).run(3, unknown=(7,)).run(30)
    assert tl.m.state == TableState.FREE
    assert tl.events == []


def test_passerby_through_occupied_table_is_not_a_visit():
    tl = seated(Timeline(machine())).run(2, guests=G, unknown=(7,)).run(30, guests=G)
    assert tl.m.state == TableState.SEATED_WAITING
    assert "served" not in tl.types()


def test_track_id_switch_does_not_reset_timers():
    tl = Timeline(machine()).run(20).run(10, guests=(1,)).run(10, guests=(31,)).run(10, guests=(77,))
    assert tl.m.state == TableState.SEATED_WAITING
    assert tl.of("seated")[0].ts == pytest.approx(20.0, abs=0.3)
    assert tl.m.snapshot()["wait_s"] == pytest.approx(30.0, abs=0.5)


# ---------------------------------------------------------------- визиты персонала
def test_waiter_2s_is_not_a_visit():
    tl = seated(Timeline(machine())).run(2, guests=G, staff=W).run(30, guests=G)
    assert tl.m.state == TableState.SEATED_WAITING
    assert "served" not in tl.types()


def test_waiter_6s_is_a_visit():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W)
    assert tl.m.state == TableState.SERVED
    assert tl.of("served")[0].payload["confidence"] == "high"


def test_unknown_standing_visitor_counts_with_low_confidence():
    tl = seated(Timeline(machine())).run(10, guests=G).run(8, guests=G, unknown=(9,))
    assert tl.m.state == TableState.SEATED_WAITING  # пока не ушёл — не знаем, визит ли это
    tl.run(5, guests=G)
    assert tl.m.state == TableState.SERVED
    ev = tl.of("served")[0]
    assert ev.payload["confidence"] == "low"
    assert ev.payload["response_time_s"] == pytest.approx(30.0, abs=1.0)  # до начала визита


def test_unknown_who_becomes_guest_is_not_a_visit():
    tl = seated(Timeline(machine())).run(10, guests=G, unknown=(9,))  # подсел опоздавший друг
    tl.run(30, guests=(*G, 9))
    assert tl.m.state == TableState.SEATED_WAITING
    assert "served" not in tl.types()


def test_unknown_staying_longer_than_visit_max_is_not_a_visit():
    tl = seated(Timeline(machine())).run(100, guests=G, unknown=(9,)).run(10, guests=G)
    assert "served" not in tl.types()


def test_visit_with_detection_gap_is_one_visit():
    tl = seated(Timeline(machine())).run(8, guests=G, staff=W)
    tl.run(60, guests=G)
    tl.run(3, guests=G, staff=W).run(1, guests=G).run(3, guests=G, staff=W).run(20, guests=G)
    visits = tl.of("staff_visit")
    assert len(visits) == 1
    assert visits[0].payload["duration_s"] == pytest.approx(7.0, abs=0.5)


def test_two_visits_are_counted():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W)
    for _ in range(2):
        tl.run(60, guests=G).run(6, guests=G, staff=W)
    tl.run(10, guests=G)
    assert [e.payload["n"] for e in tl.of("staff_visit")] == [2, 3]
    assert tl.m.snapshot()["visits"] == 3


# ---------------------------------------------------------------- уход
def test_walkout_before_service():
    tl = seated(Timeline(machine())).run(100, guests=G).run(100)
    assert tl.m.state == TableState.FREE
    ev = tl.of("walkout_before_service")[0]
    assert ev.payload["waited_s"] == pytest.approx(120.0, abs=1.0)
    assert "left" not in tl.types() and "cleaned" not in tl.types()


def test_seated_at_dirty_table():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W).run(60, guests=G).run(120)
    assert tl.m.state == TableState.NEEDS_CLEANING
    tl.run(30, guests=(5, 6))
    assert tl.m.state == TableState.SEATED_WAITING
    ev = tl.of("seated_at_dirty_table")[0]
    assert ev.payload["dirty_time_s"] == pytest.approx(120.0, abs=1.5)
    assert len(tl.of("seated")) == 2
    assert "cleaned" not in tl.types()


def test_two_parties_in_a_row_without_cleaning():
    tl = Timeline(machine()).run(20)
    for party in ((1, 2), (3, 4)):
        tl.run(20, guests=party).run(6, guests=party, staff=W).run(60, guests=party).run(100)
    assert tl.m.state == TableState.NEEDS_CLEANING
    assert tl.types() == ["seated", "served", "left", "seated_at_dirty_table", "seated", "served", "left"]
    sids = [e.payload["sid"] for e in tl.of("seated")]
    assert sids == [1, 2]


def test_short_staff_pass_does_not_clean():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W).run(30, guests=G).run(100)
    tl.run(4, staff=W).run(20)  # прошёл мимо грязного стола
    assert tl.m.state == TableState.NEEDS_CLEANING


def test_cleaning_during_leave_confirmation_window():
    """Официант убрал стол сразу после ухода гостей — раньше, чем прошло leave_min_s."""
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W).run(60, guests=G)
    tl.run(10).run(15, staff=W).run(80)
    assert tl.m.state == TableState.FREE
    assert tl.types() == ["seated", "served", "left", "cleaned"]
    assert tl.of("cleaned")[0].payload["cleanup_time_s"] == pytest.approx(25.0, abs=1.5)


def test_visit_while_guests_away_is_a_visit_if_they_return():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W).run(60, guests=G)
    tl.run(10).run(12, staff=W).run(20).run(30, guests=G)  # гости выходили, официант подходил
    assert tl.m.state == TableState.SERVED
    assert len(tl.of("staff_visit")) == 1 and "cleaned" not in tl.types()


def test_cleaning_by_unknown_low_confidence():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W).run(30, guests=G).run(100)
    tl.run(14, unknown=(70,)).run(5)
    assert tl.m.state == TableState.FREE
    assert tl.of("cleaned")[0].payload["confidence"] == "low"


# ---------------------------------------------------------------- старт и неизвестные
def test_preexisting_guests_are_assumed_served():
    tl = Timeline(machine()).run(30, guests=G)
    assert tl.m.state == TableState.SERVED
    assert tl.of("seated")[0].payload["preexisting"] is True
    assert "served" not in tl.types()  # время подхода для них не измеряется


def test_preexisting_grace_default():
    """По умолчанию посадка в первые 30 с потока считается «сидели до старта» (детектор мог опоздать)."""
    m = TableMachine("T01", RestaurantThresholds(), AlertThresholds())
    tl = Timeline(m).run(12).run(30, guests=G)
    assert m.state == TableState.SERVED and tl.of("seated")[0].payload["preexisting"] is True
    m2 = TableMachine("T01", RestaurantThresholds(), AlertThresholds())
    tl2 = Timeline(m2).run(40).run(30, guests=G)
    assert m2.state == TableState.SEATED_WAITING and tl2.of("seated")[0].payload["preexisting"] is False


def test_unknown_only_presence_seats_after_fallback():
    tl = Timeline(machine()).run(20).run(30, unknown=(4,))
    assert tl.m.state == TableState.FREE  # 30 с < 15*3
    tl.run(20, unknown=(4,))
    assert tl.m.state == TableState.SEATED_WAITING
    assert tl.of("seated")[0].payload["confidence"] == "low"


def test_staff_alone_never_seats():
    tl = Timeline(machine()).run(20).run(120, staff=W)
    assert tl.m.state == TableState.FREE and tl.events == []


# ---------------------------------------------------------------- алерты
def test_no_waiter_alert_fires_once():
    tl = seated(Timeline(machine())).run(200, guests=G)
    assert tl.alerts() == ["no_waiter"]
    assert tl.m.snapshot()["alert"] == "no_waiter"


def test_dirty_table_alert():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W).run(30, guests=G).run(420)
    assert tl.alerts() == ["dirty_table"]


def test_long_service_without_visit_alert_rearms_after_visit():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W).run(950, guests=G)
    assert tl.alerts() == ["long_service_without_visit"]
    tl.run(6, guests=G, staff=W).run(950, guests=G)
    assert tl.alerts() == ["long_service_without_visit"] * 2


def test_custom_thresholds_from_config():
    tl = Timeline(machine(seat_min_s=5, leave_min_s=20)).run(10).run(8, guests=G)
    assert tl.m.state == TableState.SEATED_WAITING
    tl.run(25)
    assert tl.m.state == TableState.FREE


# ---------------------------------------------------------------- эталон пустого стола
def test_appearance_signal_cleans_table():
    tl = seated(Timeline(machine(use_table_appearance=True))).run(6, guests=G, staff=W).run(30, guests=G)
    tl.run(120, table_dirty=True)
    assert tl.m.state == TableState.NEEDS_CLEANING
    tl.run(12, table_dirty=False)
    assert tl.m.state == TableState.FREE
    assert tl.of("cleaned")[0].payload["by"] == "appearance"


def test_appearance_signal_ignored_when_flag_off():
    tl = seated(Timeline(machine())).run(6, guests=G, staff=W).run(30, guests=G)
    tl.run(120, table_dirty=True).run(30, table_dirty=False)
    assert tl.m.state == TableState.NEEDS_CLEANING
