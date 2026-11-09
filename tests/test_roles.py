"""Эвристики ролей на синтетике (TASKS M4)."""
import numpy as np

from qonaq.config import RolesConfig
from qonaq.roles.tracker import GUEST, STAFF, UNKNOWN, PersonObs, RoleTracker
from qonaq.roles.uniform import HIST_SIZE, UniformProfile, similarity, torso_hist

SEATED_BOX = (100.0, 100.0, 160.0, 180.0)  # сидящий: невысокий bbox
STANDING_BOX = (300.0, 50.0, 350.0, 220.0)


def run(rt, seconds, make_people, t0=0.0, fps=5.0):
    last = {}
    n = int(seconds * fps)
    for i in range(n):
        ts = t0 + i / fps
        last = rt.update(ts, make_people(ts))
    return last, t0 + n / fps


def test_seated_still_person_becomes_guest():
    rt = RoleTracker(RolesConfig(), seat_min_s=15)
    d, _ = run(rt, 10, lambda ts: [PersonObs(1, SEATED_BOX, "T01")])
    assert d[1].role == UNKNOWN
    d, _ = run(rt, 15, lambda ts: [PersonObs(1, SEATED_BOX, "T01")], t0=10)
    assert d[1].role == GUEST and d[1].confidence >= 0.7


def test_jitter_does_not_prevent_guest():
    rng = np.random.default_rng(0)

    def people(ts):
        dx, dy = rng.normal(0, 3, 2)
        x1, y1, x2, y2 = SEATED_BOX
        return [PersonObs(1, (x1 + dx, y1 + dy, x2 + dx, y2 + dy), "T01")]

    d, _ = run(RoleTracker(seat_min_s=15), 30, people)
    assert d[1].role == GUEST


def test_short_visit_stays_unknown():
    rt = RoleTracker(seat_min_s=15)
    d, _ = run(rt, 8, lambda ts: [PersonObs(5, STANDING_BOX, "T01")])
    assert d[5].role == UNKNOWN and d[5].standing


def test_moving_person_in_zone_is_not_guest():
    def people(ts):
        x = 100 + 40 * ts  # идёт через кадр
        return [PersonObs(2, (x, 50, x + 50, 220), "T01")]

    d, _ = run(RoleTracker(seat_min_s=15), 25, people)
    assert d[2].role != GUEST


def test_multi_table_track_becomes_staff():
    rt = RoleTracker(seat_min_s=15)

    def people(ts):
        zone = ["T01", "T02", "T03", None][int(ts // 6) % 4]
        x = 100 + 150 * (int(ts // 6) % 4)
        return [PersonObs(9, (x, 50, x + 50, 220), zone)]

    d, _ = run(rt, 30, people)
    assert d[9].role == STAFF
    assert "multi_table" in d[9].reasons


def test_role_does_not_flip_on_single_frames():
    """Гость остаётся гостем, даже если встал и пошёл на пару секунд (гистерезис)."""
    rt = RoleTracker(seat_min_s=15)
    _, t = run(rt, 30, lambda ts: [PersonObs(1, SEATED_BOX, "T01")])
    roles = []
    for i in range(10):  # 2 секунды вне зоны
        d = rt.update(t + i * 0.2, [PersonObs(1, (400 + i * 20, 50, 450 + i * 20, 220), None)])
        roles.append(d[1].role)
    assert set(roles) == {GUEST}


def test_new_id_inherits_role_of_lost_track_nearby():
    """ByteTrack сменил id сидящему гостю — роль и dwell наследуются (SPEC §9a п.1)."""
    rt = RoleTracker(seat_min_s=15)
    _, t = run(rt, 30, lambda ts: [PersonObs(1, SEATED_BOX, "T01")])
    d = rt.update(t + 1.0, [PersonObs(42, SEATED_BOX, "T01")])
    assert d[42].role == GUEST


def test_new_id_far_away_does_not_inherit():
    rt = RoleTracker(seat_min_s=15)
    _, t = run(rt, 30, lambda ts: [PersonObs(1, SEATED_BOX, "T01")])
    d = rt.update(t + 1.0, [PersonObs(42, (800, 400, 860, 480), "T05")])
    assert d[42].role == UNKNOWN


def _frame_with_torso(color_bgr):
    frame = np.full((240, 320, 3), 90, np.uint8)
    frame[40:200, 100:180] = color_bgr
    return frame, (100.0, 40.0, 180.0, 200.0)


def test_torso_hist_and_similarity():
    red, box = _frame_with_torso((30, 30, 200))
    blue, _ = _frame_with_torso((200, 60, 30))
    h_red, h_red2, h_blue = torso_hist(red, box), torso_hist(red.copy(), box), torso_hist(blue, box)
    assert h_red.shape == (HIST_SIZE,) and abs(h_red.sum() - 1) < 1e-5
    assert similarity(h_red, h_red2) > 0.99
    assert similarity(h_red, h_blue) < 0.2


def test_uniform_profile_makes_staff_and_roundtrips(tmp_path):
    red, box = _frame_with_torso((30, 30, 200))
    blue, _ = _frame_with_torso((200, 60, 30))
    prof = UniformProfile([torso_hist(red, box)], threshold=0.55)
    prof = UniformProfile.load(prof.save(tmp_path / "staff.yaml"))
    rt = RoleTracker(seat_min_s=15, uniform=prof)
    d, _ = run(rt, 5, lambda ts: [
        PersonObs(1, box, "T01", torso_hist(red, box)),
        PersonObs(2, (200.0, 40.0, 280.0, 200.0), "T01", torso_hist(blue, box)),
    ])
    assert d[1].role == STAFF and "uniform" in d[1].reasons
    assert d[2].role == UNKNOWN


def test_uniform_beats_dwell_for_staff_standing_long():
    """Сотрудник в форме, долго стоящий у стола (бармен/хостес), не становится гостем."""
    red, box = _frame_with_torso((30, 30, 200))
    prof = UniformProfile([torso_hist(red, box)], threshold=0.55)
    rt = RoleTracker(seat_min_s=15, uniform=prof)
    d, _ = run(rt, 40, lambda ts: [PersonObs(1, box, "T01", torso_hist(red, box))])
    assert d[1].role == STAFF
