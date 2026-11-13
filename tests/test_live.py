"""M7/M8: алерты (очередь, ретраи, дедупликация), профиль clinic, POS, live-сервис и API на синтетике."""
import time

import numpy as np
import pytest

from qonaq.alerts.telegram import AlertDispatcher, TelegramSender, format_alert
from qonaq.config import AlertThresholds, CameraConfig, ClinicThresholds, RestaurantThresholds, RolesConfig, SiteConfig
from qonaq.detect.base import Detection, FakeDetector
from qonaq.events.store import EventStore
from qonaq.logic.base import Event, ZoneObservation
from qonaq.logic.clinic import ClinicEngine
from qonaq.logic.restaurant import TableMachine, TableState
from qonaq.metrics.compute import compute_metrics
from qonaq.pos import Check, MockPOS, POSAdapter, create_pos
from qonaq.pos.stubs import IikoPOS
from qonaq.zones.models import Zone, ZoneSet
from tests.helpers import Timeline

SQ = [(0, 0), (100, 0), (100, 100), (0, 100)]


def alert(kind="no_waiter", zone="T01", sid=1, ts=10.0, **p) -> Event:
    return Event(ts, zone, "alert", {"kind": kind, "sid": sid, "waited_s": 185.0, **p}, "cam")


# ---------------------------------------------------------------- алерты
class FlakySender:
    def __init__(self, fail_first: int = 0):
        self.fail_first, self.calls, self.sent = fail_first, 0, []

    def send(self, text: str) -> None:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ConnectionError("нет сети")
        self.sent.append(text)


def wait_for(cond, timeout=3.0):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_alert_text_is_russian_and_readable():
    assert format_alert(alert(), "Кафе", "Стол 4") == "⚠ Кафе: Стол 4: гости ждут официанта 03:05"


def test_alerts_without_token_go_to_log_and_feed(caplog):
    d = AlertDispatcher(None)
    with caplog.at_level("WARNING"):
        recs = d.handle([alert(), Event(1.0, "T01", "seated", {}, "cam")])
    assert len(recs) == 1 and len(d.feed) == 1 and "АЛЕРТ" in caplog.text
    d.close()


def test_alert_dedup_one_per_event():
    s = FlakySender()
    d = AlertDispatcher(s, sleep=lambda _: None)
    d.handle([alert(), alert(), alert(ts=11.0)])  # тот же стол/тип/сессия
    d.handle([alert(sid=2), alert(kind="dirty_table")])  # другие события
    assert wait_for(lambda: len(s.sent) == 3)
    d.close()
    assert len(d.feed) == 3


def test_alert_retries_until_network_is_back():
    s, sleeps = FlakySender(fail_first=3), []
    d = AlertDispatcher(s, retry_base_s=2, retry_max_s=5, sleep=sleeps.append)
    d.handle([alert()])
    assert wait_for(lambda: len(s.sent) == 1)
    d.close()
    assert s.calls == 4 and sleeps == [2, 4, 5]  # экспоненциально, с потолком
    assert d.feed[0].sent is True and d.failed_attempts == 3


def test_telegram_sender_http_contract():
    class Resp:
        status_code, text = 200, "ok"

    class Client:
        def __init__(self):
            self.calls = []

        def post(self, url, json):
            self.calls.append((url, json))
            return Resp()

    c = Client()
    TelegramSender("TOKEN", "42", client=c).send("привет")
    url, body = c.calls[0]
    assert url == "https://api.telegram.org/botTOKEN/sendMessage" and body["chat_id"] == "42" and body["text"] == "привет"
    Resp.status_code = 500
    with pytest.raises(RuntimeError):
        TelegramSender("TOKEN", "42", client=c).send("x")


# ---------------------------------------------------------------- POS
def test_mock_pos_and_occupied_without_check_alert(tmp_path):
    pos = MockPOS([Check("T01", 100.0, None)], table_map={"T09": "T01"})
    assert isinstance(pos, POSAdapter)
    assert pos.has_open_check("T01", 150) and not pos.has_open_check("T01", 50) and pos.has_open_check("T09", 150)
    th, al = RestaurantThresholds(preexisting_grace_s=5), AlertThresholds(occupied_without_check_s=600)
    with_check = Timeline(TableMachine("T01", th, al, pos=pos)).run(20).run(700, guests=(1,))
    no_check = Timeline(TableMachine("T02", th, al, pos=pos), zone_id="T02").run(20).run(700, guests=(1,))
    assert "occupied_without_check" not in with_check.alerts()
    assert no_check.alerts().count("occupied_without_check") == 1
    assert no_check.m.snapshot()["alert"] == "occupied_without_check"
    p = tmp_path / "pos.yaml"
    p.write_text("checks:\n  - {table: T05, opened: 10, closed: 20}\n")
    assert [c.table for c in MockPOS.from_yaml(p).open_checks(15)] == ["T05"] and MockPOS.from_yaml(p).open_checks(25) == []


def test_pos_factory_and_stubs():
    cfg = SiteConfig().pos
    assert create_pos(cfg) is None
    cfg.adapter = "mock"
    assert isinstance(create_pos(cfg), MockPOS)
    with pytest.raises(NotImplementedError, match="iiko"):
        IikoPOS().open_checks(0)


# ---------------------------------------------------------------- clinic
def clinic(staff_area=True, **al) -> ClinicEngine:
    zones = ZoneSet(zones=[
        Zone(id="W01", type="waiting_area", polygon=SQ),
        Zone(id="S01", type="service_point", polygon=SQ, staff_polygon=SQ if staff_area else None),
    ])
    site = SiteConfig(profile="clinic", clinic=ClinicThresholds(), alerts=AlertThresholds(**al))
    return ClinicEngine(zones, site)


def feed(eng, t0, seconds, waiting=(), desk=(), staff=None, fps=5.0):
    out, n = [], int(seconds * fps)
    for i in range(n):
        ts = t0 + i / fps
        out += eng.update([ZoneObservation(ts, "W01", unknown_tracks=tuple(waiting)),
                           ZoneObservation(ts, "S01", unknown_tracks=tuple(desk), staff_behind_desk=staff)])
    return out, t0 + seconds


def test_clinic_wait_then_served_at_desk():
    eng = clinic()
    ev, t = feed(eng, 0, 10, staff=True)
    e1, t = feed(eng, t, 120, waiting=(1,), staff=True)
    assert [e.type for e in e1] == ["wait_start"] and eng.snapshot(t)["W01"]["count"] == 1
    assert eng.snapshot(t)["W01"]["longest_s"] == pytest.approx(120, abs=1)
    e2, t = feed(eng, t, 40, desk=(1,), staff=True)  # подошёл к стойке
    e3, t = feed(eng, t, 20, staff=True)
    types = [(e.type, e.payload.get("outcome")) for e in e2 + e3]
    assert ("wait_end", "served") in types and ("service_start", None) in types and ("service_end", "completed") in types
    wait_end = next(e for e in e2 + e3 if e.type == "wait_end")
    assert wait_end.payload["duration_s"] == pytest.approx(120, abs=1)
    service_end = next(e for e in e2 + e3 if e.type == "service_end")
    assert service_end.payload["duration_s"] == pytest.approx(40, abs=1)


def test_clinic_left_without_service_and_passerby():
    eng = clinic()
    _, t = feed(eng, 0, 10)
    e1, t = feed(eng, t, 2, waiting=(5,))  # прошёл мимо — не ожидание
    e2, t = feed(eng, t, 60, waiting=(7,))
    e3, t = feed(eng, t, 30)
    ev = e1 + e2 + e3
    assert [e.type for e in ev] == ["wait_start", "wait_end"]
    assert ev[1].payload["outcome"] == "left" and ev[1].payload["duration_s"] == pytest.approx(60, abs=1)


def test_clinic_flicker_does_not_split_wait():
    eng = clinic()
    _, t = feed(eng, 0, 10)
    ev = []
    for _ in range(10):
        a, t = feed(eng, t, 10, waiting=(3,))
        b, t = feed(eng, t, 3)  # пропал на 3 с (< exit_min_s)
        ev += a + b
    assert [e.type for e in ev] == ["wait_start"]


def test_clinic_desk_unattended_alert_and_rearm():
    eng = clinic(desk_unattended_s=60)
    _, t = feed(eng, 0, 10, staff=True)
    e1, t = feed(eng, t, 50, desk=(1,), staff=False)
    assert not [e for e in e1 if e.type == "alert"]
    e2, t = feed(eng, t, 30, desk=(1,), staff=False)
    alerts = [e for e in e1 + e2 if e.type == "alert"]
    assert len(alerts) == 1 and alerts[0].payload["kind"] == "desk_unattended"
    assert eng.snapshot(t)["S01"]["state"] == "UNATTENDED"
    e3, t = feed(eng, t, 20, desk=(1,), staff=True)  # сотрудник вернулся
    assert eng.snapshot(t)["S01"]["state"] == "ATTENDED"
    e4, t = feed(eng, t, 90, desk=(2,), staff=False)
    assert len([e for e in e3 + e4 if e.type == "alert"]) == 1  # новый эпизод — новый алерт


def test_clinic_no_staff_area_means_no_unattended_alert():
    eng = clinic(staff_area=False, desk_unattended_s=10)
    ev, t = feed(eng, 0, 120, desk=(1,), staff=None)
    assert not [e for e in ev if e.type == "alert"] and eng.snapshot(t)["S01"]["staffed"] is None


def test_clinic_long_wait_alert_and_metrics():
    eng = clinic(long_wait_s=100)
    store = EventStore()
    _, t = feed(eng, 0, 10, staff=True)
    ev, t = feed(eng, t, 150, waiting=(1, 2), staff=True)
    ev2, t = feed(eng, t, 30, waiting=(2,), desk=(1,), staff=True)
    ev3, t = feed(eng, t, 30, staff=True)
    allev = ev + ev2 + ev3
    for e in allev:
        e.camera_id = "hall"
    store.add(allev)
    store.set_meta(time_base="video", scene_s=t)
    assert [e.payload["kind"] for e in allev if e.type == "alert"] == ["long_wait", "long_wait"]
    m = compute_metrics(store, SiteConfig(profile="clinic"))["clinic"]
    assert m["waits"]["n"] == 2 and m["served"] == 1 and m["left_without_service"] == 1
    assert m["waits"]["median"] == pytest.approx(165, abs=2)
    assert m["service"]["n"] == 1 and m["service"]["mean"] == pytest.approx(30, abs=1)


# ---------------------------------------------------------------- live-сервис + API (без модели и камеры)
class ScriptedSource:
    """Источник-заглушка: N кадров с шагом 0.2 с времени сцены."""

    def __init__(self, n=60):
        from qonaq.sources.video import SourceInfo

        self.n, self.i, self.info = n, 0, SourceInfo(kind="file", width=320, height=240, fps=5.0, frame_count=n)

    def open(self):
        return self

    def close(self):
        pass

    def read(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return np.full((240, 320, 3), 60, np.uint8), (self.i - 1) * 0.2


@pytest.fixture()
def live(tmp_path):
    from qonaq.api.service import LiveService

    site = SiteConfig(site_id="t", name="Тест", runs_dir=str(tmp_path),
                      cameras=[CameraConfig(id="cam", source="scripted.mp4")],
                      thresholds=RestaurantThresholds(seat_min_s=2, preexisting_grace_s=0, dropout_tolerance_s=1),
                      alerts=AlertThresholds(no_waiter_s=4),
                      roles=RolesConfig(standing_guest_min_s=2))  # bbox 80×160 — «стоит» (W2: стоящий = гость позже)
    zones = ZoneSet(camera_id="cam", frame_size=(320, 240),
                    zones=[Zone(id="T01", polygon=[(0, 0), (320, 0), (320, 240), (0, 240)])])
    det = FakeDetector([[Detection(100, 40, 180, 200, 0.9, "person")]])
    svc = LiveService(site, detector=det, use_env_sender=False)
    # подменяем зоны и источник на синтетику
    from qonaq.pipeline import CameraRuntime

    svc.pipeline.cameras["cam"] = CameraRuntime(site, site.cameras[0], zones)
    svc.pipeline.sources["cam"] = ScriptedSource(60)
    return svc


def test_live_service_end_to_end(live):
    live.start()
    assert live.finished.wait(20)
    state = live.state()
    live.stop()
    assert state["error"] is None and state["site"]["name"] == "Тест"
    cam = state["cameras"][0]
    assert cam["zones"][0]["zone_id"] == "T01" and cam["zones"][0]["state"] == "SEATED_WAITING"
    assert cam["summary"]["waiting"] == 1 and cam["summary"]["load_pct"] == 100
    assert [a["kind"] for a in state["alerts"]] == ["no_waiter"]  # без токена Telegram алерт в ленте
    assert state["telegram"] is False
    assert live.jpeg("cam")[:2] == b"\xff\xd8"
    store = EventStore(live.run_dir / "events.sqlite")
    types = [e["type"] for e in store.events()]
    assert "seated" in types and "alert" in types and (live.run_dir / "run.json").exists()


def test_api_endpoints(live):
    from fastapi.testclient import TestClient

    from qonaq.api.app import create_app

    live.start()
    assert live.finished.wait(20)
    client = TestClient(create_app(live))
    assert "Qonaq Vision" in client.get("/").text
    st = client.get("/api/state").json()
    assert st["cameras"][0]["camera_id"] == "cam"
    time.sleep(0.3)  # дать потоку записи сбросить очередь
    ev = client.get("/api/events?limit=10").json()
    assert ev and {"id", "ts", "zone_id", "type", "payload"} <= set(ev[0])
    assert client.get("/frame/cam.jpg").headers["content-type"] == "image/jpeg"
    assert client.get("/video/nope").status_code == 404
    live.stop()


def test_table_state_enum_is_plain_string():
    assert TableState.FREE == "FREE" and f"{TableState.SERVED}" == "SERVED"
