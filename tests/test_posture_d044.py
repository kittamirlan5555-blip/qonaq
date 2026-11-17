"""D-044: поза по ключевым точкам, занятость стола только по сидящим, маска неподвижных ложных «людей»."""
from __future__ import annotations

import numpy as np

from qonaq.detect.base import Detection
from qonaq.detect.pose import BEND, SIT, STAND, posture
from qonaq.detect.static_mask import StaticMask
from qonaq.logic.restaurant import TableMachine, TableState
from tests.helpers import Timeline


def kps(sh, hip, knee=None, conf=0.9) -> np.ndarray:
    k = np.zeros((17, 3), np.float32)
    for i in (5, 6):
        k[i] = (*sh, conf)
    for i in (11, 12):
        k[i] = (*hip, conf)
    if knee is not None:
        for i in (13, 14):
            k[i] = (*knee, conf)
    return k


# ---------------------------------------------------------------- поза
def test_standing_long_vertical_thigh():
    assert posture(kps((100, 100), (100, 200), (100, 290))) == STAND


def test_sitting_short_or_horizontal_thigh():
    assert posture(kps((100, 100), (100, 200), (110, 235))) == SIT  # бедро к камере — короткая проекция
    assert posture(kps((100, 100), (100, 200), (190, 215))) == SIT  # бедро вбок


def test_bent_over_table_is_not_sitting():
    """Официант нагнулся протереть стол: торс почти горизонтален, ноги прямые — «нагнулся», не «сидит»."""
    assert posture(kps((200, 150), (100, 200), (100, 290))) == BEND
    assert posture(kps((200, 150), (100, 200))) == BEND  # колени не видно


def test_unknown_without_shoulders_or_hips():
    assert posture(kps((100, 100), (100, 200), conf=0.1)) is None
    assert posture(kps((100, 100), (100, 200))) is None  # торс вертикален, ног не видно — не угадываем


# ---------------------------------------------------------------- маска неподвижных
def _det(box):
    return Detection(*box, 0.8, "person")


def test_static_detection_is_masked_after_static_s_and_unmasked_when_it_moves():
    m = StaticMask(static_s=60, drift=0.05)
    box = (100, 100, 160, 200)
    for t in range(0, 70):
        out = m.filter(float(t), [_det(box)])
    assert out == []  # 70 с без движения — маска
    out = m.filter(70.0, [_det((120, 110, 180, 210))])  # сдвинулся на 20 px из ~117 — снова человек
    assert len(out) == 1


def test_seated_guest_who_moves_a_bit_is_never_masked():
    m = StaticMask(static_s=60, drift=0.05)
    rng = np.random.default_rng(0)
    kept = 0
    for t in range(0, 600):
        j = rng.normal(0, 4, 4) if t % 20 else rng.normal(0, 12, 4)  # раз в 20 с — заметное движение
        kept += len(m.filter(float(t), [_det(tuple(np.add((100, 100, 160, 200), j)))]))
    assert kept == 600


# ---------------------------------------------------------------- занятость только по сидящим
def machine() -> TableMachine:
    from qonaq.config import RestaurantThresholds

    return TableMachine("T01", RestaurantThresholds(preexisting_grace_s=0))


def test_people_standing_at_table_never_open_session():
    tl = Timeline(machine()).run(120, unknown=(1, 2), standing_tracks=(1, 2), seated_tracks=())
    assert tl.m.state == TableState.FREE and "seated" not in tl.types()


def test_seated_open_session_regardless_of_role():
    tl = Timeline(machine()).run(30, unknown=(1,), standing_tracks=(), seated_tracks=(1,))
    assert tl.m.state == TableState.SEATED_WAITING
    assert tl.of("seated")[0].ts == 0.0


def test_bent_waiter_at_occupied_table_is_a_visit_not_a_guest():
    tl = Timeline(machine()).run(30, unknown=(1,), standing_tracks=(), seated_tracks=(1,))
    tl.run(12, unknown=(1, 9), standing_tracks=(9,), seated_tracks=(1,))  # 9 нагнулся (bend → стоящие)
    tl.run(10, unknown=(1,), standing_tracks=(), seated_tracks=(1,))
    sv = tl.of("served")
    assert len(sv) == 1 and sv[0].ts == 30.0


def test_left_when_no_one_seated_for_90s_even_with_crowd_standing_around():
    """Hog's Breath: после ухода компании у стола непрерывно стоят и проходят люди — стол всё равно «ушли»."""
    tl = Timeline(machine()).run(30, unknown=(1, 2), standing_tracks=(), seated_tracks=(1, 2))
    tl.run(8, unknown=(1, 2, 9), standing_tracks=(9,), seated_tracks=(1, 2))  # официант
    tl.run(22, unknown=(1, 2), standing_tracks=(), seated_tracks=(1, 2))
    tl.run(10, unknown=(5,), standing_tracks=(5,), seated_tracks=())
    for k in range(12):
        tl.run(10, unknown=(100 + k,), standing_tracks=(100 + k,), seated_tracks=())
    left = tl.of("left")
    assert left and abs(left[0].ts - 60.0) <= 1.0


def test_guest_who_sat_and_stands_up_is_not_a_visit():
    tl = Timeline(machine()).run(30, unknown=(1, 2), standing_tracks=(), seated_tracks=(1, 2))
    tl.run(20, unknown=(1, 2), standing_tracks=(2,), seated_tracks=(1,))  # 2 встал, поговорить стоя
    tl.run(10, unknown=(1, 2), standing_tracks=(), seated_tracks=(1, 2))
    assert "served" not in tl.types() and "staff_visit" not in tl.types()


# ---------------------------------------------------------------- D-045: «убрано»
def _left_tl() -> Timeline:
    tl = Timeline(machine()).run(30, unknown=(1,), standing_tracks=(), seated_tracks=(1,))
    tl.run(8, unknown=(1, 9), standing_tracks=(9,), seated_tracks=(1,))  # официант
    tl.run(20, unknown=(1,), standing_tracks=(), seated_tracks=(1,))
    tl.run(100, standing_tracks=(), seated_tracks=())  # ушли
    assert tl.m.state == TableState.NEEDS_CLEANING
    return tl


def test_cleaning_is_a_standing_or_bent_visit_of_5s_after_left():
    tl = _left_tl()
    tl.run(6, unknown=(20,), standing_tracks=(20,), seated_tracks=()).run(5, standing_tracks=(), seated_tracks=())
    cl = tl.of("cleaned")
    assert cl and tl.m.state == TableState.FREE and abs(cl[0].ts - (158 + 6)) <= 1.0


def test_friend_who_comes_to_empty_table_and_sits_is_not_cleaning():
    tl = _left_tl()
    tl.run(6, unknown=(30,), standing_tracks=(30,), seated_tracks=())  # подошёл к пустому столу
    tl.run(30, unknown=(30,), standing_tracks=(), seated_tracks=(30,))  # и сел
    assert "cleaned" not in tl.types()
    assert "seated" in tl.types()[1:] and tl.m.state != TableState.FREE


def test_left_time_is_when_last_sitter_walked_away_not_when_stood_up():
    """D-046: встали в 60 с, постояли у стола 30 с, ушли в 90 с — «ушли» = 90 с; подтверждение — 90 с без сидящих."""
    tl = Timeline(machine()).run(30, unknown=(1, 2), standing_tracks=(), seated_tracks=(1, 2))
    tl.run(8, unknown=(1, 2, 9), standing_tracks=(9,), seated_tracks=(1, 2))
    tl.run(22, unknown=(1, 2), standing_tracks=(), seated_tracks=(1, 2))
    tl.run(30, unknown=(1, 2), standing_tracks=(1, 2), seated_tracks=())
    tl.run(120, standing_tracks=(), seated_tracks=())
    left = tl.of("left")
    assert left and abs(left[0].ts - 90.0) <= 1.0
    assert left[0].payload["detected_ts"] <= 60.0 + 90.0 + 6.0  # подтверждено через ~90 с после того, как встали
