"""Сквозной тест пайплайна без модели и без видео: сценарный детектор -> трекер -> зоны -> роли ->
логика -> SQLite -> метрики. Проверяет длинный сценарий (уход через 90 с), которого нет в коротких роликах."""
import json

import numpy as np
import pytest

from qonaq.config import CameraConfig, SiteConfig
from qonaq.detect.base import Detection
from qonaq.events.store import EventStore
from qonaq.metrics.compute import compute_metrics
from qonaq.pipeline import CameraRuntime, Pipeline
from qonaq.render.annotate import DemoRenderer, render_tracks
from qonaq.roles.calibrate import build_profile
from qonaq.roles.uniform import HIST_SIZE
from qonaq.sources.video import SourceInfo
from qonaq.zones.models import Zone, ZoneSet

FPS = 5.0
GUEST_A, GUEST_B = (120, 120, 180, 210), (230, 120, 290, 210)  # сидят за T01
WAITER = (175, 60, 225, 230)  # стоит у T01 (h/w = 3.4)
OTHER = (480, 120, 540, 210)  # гость за T02


def person(box, jitter=0.0, rng=None):
    j = rng.normal(0, jitter, 4) if rng is not None and jitter else np.zeros(4)
    return Detection(*(np.array(box, float) + j), 0.9, "person")


class Scenario:
    """Источник + детектор по сценарию: функция времени -> список рамок людей."""

    name, device = "scripted", "cpu"

    def __init__(self, duration_s, people_at):
        self.n, self.i, self.people_at = int(duration_s * FPS), 0, people_at
        self.info = SourceInfo(kind="file", width=640, height=360, fps=FPS, frame_count=self.n)
        self.rng = np.random.default_rng(7)
        self._ts = 0.0

    def open(self):
        return self

    def close(self):
        pass

    def read(self):
        if self.i >= self.n:
            return None
        self._ts = self.i / FPS
        self.i += 1
        return np.full((360, 640, 3), 70, np.uint8), self._ts

    def detect(self, frames):
        return [[person(b, 1.5, self.rng) for b in self.people_at(self._ts)] for _ in frames]


ZONES = ZoneSet(camera_id="cam", frame_size=(640, 360), zones=[
    Zone(id="T01", polygon=[(90, 90), (320, 90), (320, 250), (90, 250)]),
    Zone(id="T02", polygon=[(440, 90), (600, 90), (600, 250), (440, 250)]),
])


def full_cycle(t):
    """T01: сели в 40 с, официант 100–108, ушли в 300, уборка 420–435. T02: гость сидит с 0 до конца."""
    boxes = [OTHER]
    if 40 <= t < 300:
        boxes += [GUEST_A, GUEST_B]
    if 100 <= t < 108 or 420 <= t < 435:
        boxes.append(WAITER)
    return boxes


@pytest.fixture(scope="module")
def cycle_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    site = SiteConfig(site_id="t", name="Сценарий", cameras=[CameraConfig(id="cam", source="x.mp4")])
    sc = Scenario(520, full_cycle)
    store = EventStore(tmp / "events.sqlite")
    pipe = Pipeline(site, detector=sc, store=store, zones={"cam": ZONES}, sources={"cam": sc})
    frames = []
    stats = pipe.run(on_result=lambda r: frames.append(r) if r.frame_index in (300, 1000, 2000) else None)
    store.set_meta(time_base="video", scene_s=stats.scene_s, tables=2)
    return site, store, frames, stats, pipe


def test_full_table_cycle_through_whole_pipeline(cycle_run):
    _, store, _, stats, _ = cycle_run
    ev = [(e["type"], e["zone_id"], round(e["ts"])) for e in store.events() if e["type"] != "state_change"]
    types_t1 = [t for t, z, _ in ev if z == "T01"]
    assert types_t1 == ["seated", "served", "left", "cleaned"]
    by = {t: ts for t, z, ts in ev if z == "T01"}
    assert by["seated"] == pytest.approx(40, abs=2)
    assert by["served"] == pytest.approx(100, abs=2)
    assert by["left"] == pytest.approx(300, abs=2)  # задним числом, хотя подтверждён только через 90 с
    assert by["cleaned"] == pytest.approx(435, abs=3)
    assert ("seated", "T02", 0) in [(t, z, 0) for t, z, _ in ev]  # сидел с самого начала
    assert stats.frames == 2600


def test_sessions_and_metrics_from_pipeline(cycle_run):
    site, store, *_ = cycle_run
    s1 = next(s for s in store.sessions() if s["zone_id"] == "T01")
    assert s1["outcome"] == "completed" and not s1["preexisting"]
    assert s1["response_time_s"] == pytest.approx(60, abs=2.5)
    assert s1["visit_duration_s"] == pytest.approx(260, abs=3)
    assert s1["cleanup_time_s"] == pytest.approx(135, abs=4)
    s2 = next(s for s in store.sessions() if s["zone_id"] == "T02")
    assert s2["preexisting"] == 1 and s2["outcome"] == "open"
    m = compute_metrics(store, site)
    assert m["response"]["n"] == 1 and m["response"]["median"] == pytest.approx(60, abs=2.5)
    assert m["dirty"]["mean"] == pytest.approx(135, abs=4)
    # пик = оба стола заняты (40..300 с): грязный стол уже вне пика => потерь нет (консервативная оценка)
    assert m["roi"]["peak_s"] == pytest.approx(260, abs=4) and m["roi"]["loss_kzt"] == 0


def test_renderers_produce_frames_and_blur(cycle_run):
    site, _, frames, _, pipe = cycle_run
    r = frames[1]  # t=200 с: гости сидят
    r.frame[:] = np.random.default_rng(0).integers(0, 255, r.frame.shape, dtype=np.uint8)  # чтобы блюр был заметен
    demo = DemoRenderer(ZONES, site).render(r)
    assert demo.shape[0] == 720 and demo.shape[1] > 1280  # кадр + панель
    plain = DemoRenderer(ZONES, site, blur=False).render(r)
    assert not np.array_equal(demo, plain)
    tr = render_tracks(r, ZONES)
    assert tr.shape == r.frame.shape and not np.array_equal(tr, r.frame)
    assert {p.role for p in r.people} == {"guest"}
    assert r.summary["load_pct"] == 100 and r.zone_states["T01"]["state"] == "SERVED"


def test_no_zones_still_tracks():
    site = SiteConfig(cameras=[CameraConfig(id="cam", source="x.mp4")])
    sc = Scenario(4, lambda t: [GUEST_A])
    cam = CameraRuntime(site, site.cameras[0], ZoneSet(camera_id="cam"))
    out = None
    while (item := sc.read()) is not None:
        out = cam.process(item[0], item[1], sc.detect([item[0]])[0])
    assert len(out.people) == 1 and out.zone_states == {} and out.events == []


def test_low_confidence_detections_are_blurred_but_not_tracked():
    site = SiteConfig(cameras=[CameraConfig(id="cam", source="x.mp4")])
    cam = CameraRuntime(site, site.cameras[0], ZONES)
    frame = np.zeros((360, 640, 3), np.uint8)
    res = None
    for i in range(5):
        res = cam.process(frame, i * 0.2, [Detection(100, 100, 160, 220, 0.15, "person")])
    assert res.people == [] and len(res.blur_boxes) == 1


def test_calibration_profile_from_tracks_json(tmp_path):
    hist = [1.0 / HIST_SIZE] * HIST_SIZE
    (tmp_path / "tracks.json").write_text(json.dumps({
        "3": {"hist": hist, "tables": ["T01", "T02", "T03"]}, "5": {"hist": hist, "tables": ["T01"]}}))
    prof, ids = build_profile(tmp_path, [5])
    assert ids == [5] and len(prof.references) == 1
    prof, ids = build_profile(tmp_path, auto=True)
    assert ids == [3]
    with pytest.raises(ValueError):
        build_profile(tmp_path, [99])
    with pytest.raises(ValueError):
        build_profile(tmp_path)


def test_role_classifier_in_pipeline_marks_waiter_staff_early(tmp_path):
    """W2: внешний вид «персонал» у официанта → визит засчитан как подтверждённый персонал (high), а не
    «неизвестный стоя» (low); вызовов классификатора — не чаще appearance_every_s на трек."""
    from qonaq.roles.classifier import FakeRoleClassifier

    calls = []

    def fn(box):
        calls.append(box)
        return 0.95 if (box[3] - box[1]) / (box[2] - box[0]) > 2.5 else 0.05  # высокий bbox — официант

    site = SiteConfig(site_id="t", cameras=[CameraConfig(id="cam", source="x.mp4")])
    sc = Scenario(200, lambda t: ([GUEST_A, GUEST_B] if t >= 40 else []) + ([WAITER] if 100 <= t < 108 else []))
    store = EventStore(tmp_path / "events.sqlite")
    pipe = Pipeline(site, detector=sc, store=store, zones={"cam": ZONES}, sources={"cam": sc},
                    role_classifier=FakeRoleClassifier(fn))
    pipe.run()
    served = [e for e in store.events() if e["type"] == "served"]
    assert len(served) == 1 and served[0]["ts"] == pytest.approx(100, abs=2)
    assert served[0]["payload"].get("confidence") == "high"
    assert len(calls) < 200 * 3 / site.roles.appearance_every_s * 1.2  # ≈ 1 раз в секунду на трек, не каждый кадр
