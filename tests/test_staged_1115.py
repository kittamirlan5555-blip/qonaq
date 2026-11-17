"""Постановка 15.11 (IMG_2972, 1 стол, гость в голубом, «официант» в чёрном, камера высоко, кадр вертикальный).

Найдено на записи и исправлено (D-043, EVAL §25):
1. Подмена id трекером: прохожий/официант заслоняет сидящего, ByteTrack отдаёт ему id гостя (форма рамки почти та
   же), гость получает новый id без истории — «неизвестный стоящий у стола» 60 с = ложный визит персонала.
2. Неизвестный, сидящий у занятого стола, или вставший на место только что пропавшего сидящего, — это гость,
   а не визит; визит — только «лишний» человек у стола.
3. Порог «стоит» под камеру: с высокой камеры у сидящего видны ноги, рамка «стоячая» (2,15 при пороге 1,9).
4. Сессия не начинается от людей, которые ходят у свободного стола и не садятся (подготовка 0:12–1:20).

Синтетические таймлайны — на каждое правило; регрессия — на записи-повторе этого видео (детекции + гистограммы
торса, без кадров: tests/fixtures/replay/staged_1115_4fps.jsonl.gz) против ручной разметки владельца
(tests/fixtures/gt/staged_1115.yaml).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qonaq.config import AlertThresholds, RestaurantThresholds, RolesConfig
from qonaq.logic.restaurant import TableMachine, TableState
from qonaq.roles.tracker import GUEST, UNKNOWN, PersonObs, RoleTracker
from qonaq.roles.uniform import HIST_SIZE
from tests.helpers import Timeline

FIX = Path(__file__).parent / "fixtures"
REPLAY = FIX / "replay" / "staged_1115_4fps.jsonl.gz"
REPLAY_10 = FIX / "replay" / "staged_1115_10fps.jsonl.gz"  # профиль GPU (process_fps: auto)
GT = FIX / "gt" / "staged_1115.yaml"
ZONES = FIX / "zones" / "staged_1115.yaml"

# Рамки с этой записи (1080×1920): сидящий гость с высокой камеры — «стоячие» пропорции 2,15.
GUEST_BOX = (345.0, 1286.0, 545.0, 1715.0)
WAITER_BOX = (433.0, 1222.0, 674.0, 1909.0)  # стоит у стола, высота 687 против 429 у сидящего
PASSER_BOX = (411.0, 1288.0, 546.0, 1616.0)  # прохожий заслонил гостя: пропорции 2,43 — «как у гостя»


def _hist(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.random(HIST_SIZE).astype(np.float32) ** 4
    return h / h.sum()


BLUE, BLACK = _hist(11), _hist(22)  # голубое поло гостя / чёрная футболка официанта


def _feed(rt: RoleTracker, t0: float, seconds: float, people, fps: float = 4.0):
    d, n = {}, int(round(seconds * fps))
    for i in range(n):
        d = rt.update(t0 + i / fps, people(t0 + i / fps))
    return d, t0 + n / fps


def _seated_guest(rt: RoleTracker, seconds: float = 90.0):
    return _feed(rt, 0.0, seconds, lambda ts: [PersonObs(1, GUEST_BOX, "T01", BLUE)])


# ---------------------------------------------------------------- 1. подмена id трекером
def test_id_handed_to_passer_with_same_box_shape_does_not_steal_guest_history():
    """3:32 записи: прохожий заслонил гостя, ByteTrack отдал ему id 1, гость получил id 3. Форма рамки почти та же
    (старое правило «резкая смена формы И одежды» молчало). Одежда другая дольше 0,7 с — история гостя уходит
    с id 1 и достаётся новому id на старом месте в той же одежде."""
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    d, t = _seated_guest(rt)
    assert d[1].role == GUEST
    sid = d[1].stable_id
    d, t = _feed(rt, t, 1.0, lambda ts: [PersonObs(1, PASSER_BOX, "T01", BLACK)])  # id 1 уже на прохожем
    d, t = _feed(rt, t, 3.0, lambda ts: [PersonObs(1, (700.0, 1200.0, 950.0, 1900.0), None, BLACK),
                                         PersonObs(3, GUEST_BOX, "T01", BLUE)])
    assert d[3].role == GUEST and d[3].stable_id == sid, "гость под новым id сохранил историю"
    assert d[1].role == UNKNOWN and d[1].stable_id != sid, "прохожий не получил «сидел 90 с»"


def test_short_clothes_noise_does_not_detach_history():
    """Официант стоит перед гостем: торс гостя на кадре-другом «не того цвета» — это не подмена."""
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    d, t = _seated_guest(rt)
    sid = d[1].stable_id
    d, t = _feed(rt, t, 0.5, lambda ts: [PersonObs(1, GUEST_BOX, "T01", BLACK)])  # 2 кадра шума
    d, t = _feed(rt, t, 2.0, lambda ts: [PersonObs(1, GUEST_BOX, "T01", BLUE)])
    assert d[1].stable_id == sid and d[1].role == GUEST


def test_same_clothes_after_long_occlusion_inherits_history():
    """6:40 записи: официант закрыл гостя собой на 5 с (больше старого окна 4 с) — гость вернулся под новым id."""
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    d, t = _seated_guest(rt)
    sid = d[1].stable_id
    d, t = _feed(rt, t, 5.0, lambda ts: [PersonObs(14, WAITER_BOX, "T01", BLACK)])
    d, t = _feed(rt, t, 1.0, lambda ts: [PersonObs(14, WAITER_BOX, "T01", BLACK),
                                         PersonObs(15, (372.0, 1316.0, 468.0, 1472.0), "T01", BLUE)])
    assert d[15].stable_id == sid and d[15].role == GUEST
    assert d[14].stable_id != sid


def test_other_clothes_on_guest_place_never_inherits():
    """Официант встал ровно на место ушедшего гостя через секунду — история гостя ему не достаётся."""
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    d, t = _seated_guest(rt)
    d, t = _feed(rt, t + 1.0, 1.0, lambda ts: [PersonObs(7, GUEST_BOX, "T01", BLACK)])
    assert d[7].role == UNKNOWN and d[7].dwell_s < 5


def test_without_clothes_old_short_gap_rule_still_works():
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    _feed(rt, 0, 90, lambda ts: [PersonObs(1, GUEST_BOX, "T01")])
    assert rt.update(91.0, [PersonObs(42, GUEST_BOX, "T01")])[42].role == GUEST  # 1 с — наследует
    rt2 = RoleTracker(RolesConfig(), seat_min_s=15)
    _feed(rt2, 0, 90, lambda ts: [PersonObs(1, GUEST_BOX, "T01")])
    assert rt2.update(100.0, [PersonObs(42, GUEST_BOX, "T01")])[42].role == UNKNOWN  # 10 с без одежды — нет


# ---------------------------------------------------------------- 3. порог «стоит» под камеру
def test_standing_threshold_learns_seated_height_of_this_table():
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    d, t = _feed(rt, 0, 5, lambda ts: [PersonObs(1, GUEST_BOX, "T01", BLUE)])
    assert d[1].standing, "пока эталона нет — по пропорциям: 2,15 ≥ 1,9 — «стоит»"
    d, t = _feed(rt, t, 80, lambda ts: [PersonObs(1, GUEST_BOX, "T01", BLUE)])
    assert d[1].role == GUEST and rt.seated_height("T01") == pytest.approx(429, abs=2)
    d, t = _feed(rt, t, 1, lambda ts: [PersonObs(1, GUEST_BOX, "T01", BLUE),
                                       PersonObs(2, WAITER_BOX, "T01", BLACK)])
    assert not d[1].standing and d[2].standing
    assert rt.seated_height("T02") is None, "эталон — по столу, у другого стола свой"


def test_standing_guest_at_bar_is_not_learned_as_seated():
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    tall = (300.0, 900.0, 400.0, 1250.0)  # 3,5: стоит у барной стойки
    d, _ = _feed(rt, 0, 100, lambda ts: [PersonObs(1, tall, "B1", BLUE)])
    assert d[1].role == GUEST and rt.seated_height("B1") is None


# ---------------------------------------------------------------- 4. нагнулся ≠ сел
def test_bending_over_table_for_seconds_does_not_make_guest():
    """Подготовка 0:57–1:01: человек протирает стол, рамка на секунды «сидячая» — гостем он не становится."""
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    bent, up = (330.0, 1300.0, 700.0, 1800.0), (430.0, 1220.0, 700.0, 1900.0)

    def people(ts):
        return [PersonObs(5, bent if int(ts) % 10 < 5 else up, "T01", BLACK)]  # 5 с нагнулся, 5 с выпрямился

    d, _ = _feed(rt, 0, 60, people)
    assert d[5].role != GUEST


def test_really_seated_unknown_becomes_guest_after_seat_min():
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    seated = (100.0, 100.0, 160.0, 180.0)  # 1,33 — сидит по пропорциям
    d, _ = _feed(rt, 0, 25, lambda ts: [PersonObs(1, seated, "T01")])
    assert d[1].role == GUEST


# ---------------------------------------------------------------- 2. «за столом» против визита (логика стола)
G = (1,)


def machine(**th) -> TableMachine:
    th.setdefault("preexisting_grace_s", 0.0)
    return TableMachine("T01", RestaurantThresholds(**th), AlertThresholds())


def seated_tl() -> Timeline:
    return Timeline(machine(), fps=4.0).run(20).run(30, guests=G, standing_tracks=())


def test_guest_with_new_id_standing_box_is_not_a_visit():
    """Гость (id 1) пропал, через секунду на его месте «неизвестный стоящий» (это он, новый id) — не визит."""
    tl = seated_tl()
    tl.run(1, standing_tracks=())
    tl.run(60, unknown=(3,), standing_tracks=(3,))
    assert tl.m.state == TableState.SEATED_WAITING and "served" not in tl.types()


def test_waiter_arriving_while_guest_changes_id_is_a_visit_of_right_length():
    """Официант подошёл (id 9) и в этот же момент гость под ним сменил id (1 → 3): визит — 8 с официанта.
    D-044: новый id гостя по позе сидит — он «за столом» сам по себе, без правила наследования D-043."""
    tl = seated_tl()
    tl.run(1, guests=G, unknown=(9,), standing_tracks=(9,))
    tl.run(7, unknown=(9, 3), standing_tracks=(9,))
    tl.run(30, unknown=(3,), standing_tracks=())
    sv = tl.of("served")
    assert len(sv) == 1 and sv[0].ts == pytest.approx(50.0, abs=0.5)
    assert tl.types().count("staff_visit") == 0


def test_guest_standing_up_to_leave_after_visit_does_not_extend_visit():
    """8:02–8:18 записи: официант 10 с у стола, затем гость встал (стоит, неизвестный) и ушёл — визит 10 с,
    засчитан, а не «пропал» вместе с уходом гостя."""
    tl = seated_tl()
    tl.run(10, guests=G, staff=(50,))  # подход — served
    tl.run(60, guests=G)
    tl.run(10, guests=G, unknown=(9,), standing_tracks=(9,))  # официант
    tl.run(1, unknown=(9, 4), standing_tracks=(9,))  # гость сменил id (4 сидит)
    tl.run(4, unknown=(4,), standing_tracks=(4,))  # встал
    tl.run(5, unknown=(4,), standing_tracks=(4,))
    tl.run(10)
    visits = tl.of("staff_visit")
    assert len(visits) == 1 and visits[0].payload["duration_s"] == pytest.approx(11, abs=1.5)


def test_guest_new_id_flickering_keeps_its_place():
    """10 к/с, 8:03 записи: гость под официантом сменил id (1 → 44), id 44 на секунду пропал и вернулся — он
    по-прежнему «за столом»; визит — только официант."""
    tl = seated_tl()
    tl.run(10, guests=G, staff=(50,))
    tl.run(60, guests=G)
    tl.run(0.5, guests=G, unknown=(9,), standing_tracks=(9,))
    tl.run(0.5, unknown=(9,), standing_tracks=(9,))
    tl.run(0.25, unknown=(9, 44), standing_tracks=(9,))
    tl.run(1.0, unknown=(9,), standing_tracks=(9,))  # 44 пропал
    tl.run(8, unknown=(9, 44), standing_tracks=(9,))  # вернулся; по позе (D-044) сидит
    tl.run(10, unknown=(44,), standing_tracks=())  # официант ушёл
    tl.run(10, unknown=(44,), standing_tracks=())
    visits = tl.of("staff_visit")
    assert len(visits) == 1 and visits[0].payload["duration_s"] == pytest.approx(10.25, abs=1.0)


def test_waiter_coming_to_clean_after_guest_left_is_not_a_guest():
    """Окно «он же со сменой id» короткое: официант, пришедший через 14 с после ухода гостя, — уборка/визит,
    а не «гость вернулся»; стол уходит в «ушли»."""
    tl = seated_tl()
    tl.run(8, guests=G, staff=(50,))
    tl.run(30, guests=G)
    tl.run(14)
    tl.run(15, unknown=(9,), standing_tracks=(9,))
    tl.run(100)
    assert "left" in tl.types()
    left = tl.of("left")[0]
    assert left.ts == pytest.approx(20 + 30 + 8 + 30, abs=1.5)


def test_people_walking_around_free_table_do_not_start_session():
    """Подготовка 0:12–1:20: двое ходят у стола (стоят), не садятся — больше 45 с. Сессии нет."""
    tl = Timeline(machine(), fps=4.0).run(10)
    tl.run(70, unknown=(5, 6), standing_tracks=(5, 6))
    assert tl.m.state == TableState.FREE and "seated" not in tl.types()


def test_seated_unknowns_still_start_session_without_role():
    tl = Timeline(machine(), fps=4.0).run(10)
    tl.run(50, unknown=(5,), standing_tracks=())
    assert tl.m.state == TableState.SEATED_WAITING


# ---------------------------------------------------------------- регрессия на реальной записи
def _site():
    from qonaq.config import PROJECT_ROOT, CameraConfig, load_site

    site = load_site(PROJECT_ROOT / "configs" / "sites" / "staged.yaml")
    site.process_fps = 4.0
    site.cameras = [CameraConfig(id="staged_1115", source="replay", zones=str(ZONES))]
    return site


def _replay(start: float, path: Path = REPLAY):
    from qonaq.detect.replay import run

    return [e for e in run(path, _site(), start_s=start) if e.type != "state_change"]


def _check_against_gt(events, start: float):
    """Критерии владельца (4 к/с, без калибровки): проход мимо в 3:32 не засчитан, подход в 5:04 — первый визит,
    визит в 6:39 длится ~8 с, визит в 8:02 пойман; посадка — в 2:16; алертов при боевых порогах нет."""
    from qonaq.evalgt import Ev, evaluate, load_gt

    gt = [Ev(g.t - start, g.zone, g.type) for g in load_gt(GT) if g.t >= start]
    sysev = [Ev(e.ts, e.zone_id, e.type) for e in events]
    res = evaluate(gt, sysev, tolerance_s=10)
    by = res["by_type"]
    assert by["seated"]["tp"] == 1 and by["seated"]["fp"] == 0
    assert by["served"]["tp"] == 1 and by["served"]["fp"] == 0
    assert by["staff_visit"]["tp"] == 2 and by["staff_visit"]["fp"] == 0
    served = next(e for e in events if e.type == "served")
    assert served.ts + start == pytest.approx(304, abs=3), "первый визит — заказ в 5:04, а не проход в 3:32"
    v639 = next(e for e in events if e.type == "staff_visit" and abs(e.ts + start - 399) < 10)
    assert 5 <= v639.payload["duration_s"] <= 12
    assert not [e for e in events if e.type == "alert"]
    return res


@pytest.mark.skipif(not REPLAY.exists(), reason="нет записи-повтора")
def test_staged_1115_cut_from_1_25():
    _check_against_gt(_replay(85.0), 85.0)


@pytest.mark.skipif(not REPLAY.exists(), reason="нет записи-повтора")
def test_staged_1115_full_file_preparation_does_not_open_session():
    events = _replay(0.0)
    assert not [e for e in events if e.type in ("seated", "served", "staff_visit") and e.ts < 130], \
        "подготовка 0:12–1:20 (двое ходят у стола) — не посадка и не визиты"
    _check_against_gt(events, 0.0)


@pytest.mark.skipif(not REPLAY_10.exists(), reason="нет записи-повтора 10 к/с")
@pytest.mark.parametrize("start", [0.0, 85.0])
def test_staged_1115_gpu_profile_10fps(start):
    _check_against_gt(_replay(start, REPLAY_10), start)


def test_process_fps_auto_gpu_10_cpu_4(tmp_path):
    """D-043: `process_fps: auto` — 10 кадр/с, если детектор реально на GPU, иначе 4 (на 4 к/с без исправлений
    гость терял id при перекрытии; исправления работают и на 4, а GPU позволяет больше)."""
    from qonaq.config import load_site
    from qonaq.detect.base import FakeDetector
    from qonaq.pipeline import Pipeline

    p = tmp_path / "s.yaml"
    p.write_text("process_fps: auto\ncameras: [{id: c, source: x.mp4}]\n", encoding="utf-8")
    assert load_site(p).fps_auto and load_site(p).resolve_fps("cuda:0") == 10
    assert load_site(p).resolve_fps("cpu") == 4
    site = load_site(p)
    Pipeline(site, detector=FakeDetector())  # device "cpu"
    assert site.process_fps == 4 and not site.fps_auto
    p.write_text("process_fps: 6\n", encoding="utf-8")
    assert load_site(p).resolve_fps("cuda:0") == 6, "явное число в конфиге не трогаем"
