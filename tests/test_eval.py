"""M8a: сравнение с разметкой и сигнал по эталону пустого стола — на синтетике."""
from pathlib import Path

import numpy as np
import pytest

from qonaq.evalgt import Ev, evaluate, format_table, load_gt, match_events, run_eval
from qonaq.events.store import EventStore
from qonaq.logic.base import Event
from qonaq.zones.appearance import TableAppearance
from qonaq.zones.models import Zone, ZoneSet

GT = [Ev(10, "T01", "seated"), Ev(70, "T01", "served"), Ev(300, "T01", "left"), Ev(20, "T02", "seated")]


def test_perfect_match_within_window():
    sys = [Ev(12, "T01", "seated"), Ev(66, "T01", "served"), Ev(309, "T01", "left"), Ev(20, "T02", "seated")]
    r = evaluate(GT, sys)
    assert r["total"]["precision"] == 1.0 and r["total"]["recall"] == 1.0
    assert r["total"]["time_error_s"] == pytest.approx((2 + 4 + 9 + 0) / 4)
    assert r["by_type"]["seated"]["tp"] == 2


def test_misses_extras_and_wrong_zone():
    sys = [Ev(25, "T01", "seated"),  # вне окна ±10 с
           Ev(70, "T02", "served"),  # не та зона
           Ev(300, "T01", "left"), Ev(305, "T01", "left")]  # дубль
    r = evaluate(GT, sys)
    assert r["by_type"]["seated"] == {"gt": 2, "system": 1, "tp": 0, "fn": 2, "fp": 1, "precision": 0.0,
                                      "recall": 0.0, "time_error_s": None}
    assert r["by_type"]["left"]["tp"] == 1 and r["by_type"]["left"]["fp"] == 1
    assert r["total"]["tp"] == 1 and len(r["missed"]) == 3 and len(r["extra"]) == 3
    assert "Пропущено" in format_table(r)


def test_matching_is_one_to_one_and_closest():
    m, missed, extra = match_events([Ev(10, "T01", "seated"), Ev(14, "T01", "seated")],
                                    [Ev(13, "T01", "seated")])
    assert len(m) == 1 and m[0][0].t == 14 and len(missed) == 1 and not extra


def test_empty_sides():
    assert evaluate([], [])["total"]["precision"] is None
    assert evaluate(GT, [])["total"]["recall"] == 0.0


def test_run_eval_from_sqlite_and_yaml(tmp_path):
    store = EventStore(tmp_path / "events.sqlite")
    store.add([Event(11.0, "T01", "seated", {"sid": 1}, "c"), Event(11.0, "T01", "state_change", {}, "c"),
               Event(72.0, "T01", "served", {"sid": 1}, "c")])
    store.close()
    gt = tmp_path / "gt.yaml"
    gt.write_text("events:\n  - {t: 10, zone: T01, event: seated}\n  - {t: 70, zone: T01, event: served}\n")
    r = run_eval(tmp_path, gt)
    assert r["total"]["tp"] == 2 and r["total"]["fp"] == 0  # state_change в сравнении не участвует
    assert (tmp_path / "eval.md").exists()


def test_committed_gt_fixtures_are_valid():
    d = Path(__file__).parent / "fixtures" / "gt"
    files = sorted(d.glob("*.yaml"))
    assert {f.stem for f in files} >= {"cctv_restaurant_corner", "cctv_terrace_cafe"}
    for f in files:
        assert load_gt(f)


# ---------------------------------------------------------------- эталон пустого стола
def _scene(dishes=False, brightness=0):
    rng = np.random.default_rng(3)
    img = np.full((200, 300, 3), 120, np.uint8) + rng.integers(0, 4, (200, 300, 3), dtype=np.uint8)
    img = np.clip(img.astype(int) + brightness, 0, 255).astype(np.uint8)
    if dishes:
        for cx, cy in [(120, 90), (170, 110), (140, 130)]:
            img[cy - 12:cy + 12, cx - 12:cx + 12] = 245
    return img


def test_appearance_reference_and_dirty_signal():
    zones = ZoneSet(zones=[Zone(id="T01", polygon=[(80, 50), (220, 50), (220, 160), (80, 160)])])
    ap = TableAppearance(zones, (300, 200))
    free_long = {"T01": {"state": "FREE", "state_s": 60}}
    dirty_state = {"T01": {"state": "NEEDS_CLEANING", "state_s": 5}}
    assert ap.update(_scene(), 0, [], dirty_state) == {"T01": None}  # эталона ещё нет
    assert ap.update(_scene(), 1, [], free_long) == {"T01": False} and ap.has_reference("T01")
    assert ap.update(_scene(dishes=True), 2, [], dirty_state) == {"T01": True}
    assert ap.update(_scene(), 3, [], dirty_state) == {"T01": False}
    # общее изменение освещения — не «посуда»
    assert ap.update(_scene(brightness=25), 4, [], dirty_state) == {"T01": False}
    # человек закрывает стол — сигнала нет
    assert ap.update(_scene(dishes=True), 5, [(100, 40, 160, 150)], dirty_state) == {"T01": None}
