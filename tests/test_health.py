"""W7: монитор здоровья, ежедневная сводка, кнопка «Неверно»; W8: уверенные алерты."""
import time

import numpy as np

from qonaq.alerts.telegram import AlertDispatcher, FeedbackPoller, TelegramSender, is_confident
from qonaq.config import AlertThresholds, RestaurantThresholds
from qonaq.events.store import EventStore
from qonaq.health.daily import FrameSampler, caption_for, due, summary_text
from qonaq.health.monitor import HealthConfig, HealthMonitor, check_events
from qonaq.logic.base import Event, ZoneObservation
from qonaq.logic.restaurant import TableMachine


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def noisy(seed, level=120):
    return np.random.default_rng(seed).integers(level - 10, level + 10, (72, 128, 3), dtype=np.uint8)


def kinds(m, live=True):
    return sorted(i.kind for i in m.check(live))


def test_camera_down_only_for_live():
    clk = Clock()
    m = HealthMonitor(HealthConfig(stale_s=30), clock=clk)
    m.observe_frame("c1", 0.0, noisy(0))
    clk.t += 31
    assert kinds(m) == ["camera_down"]
    assert kinds(m, live=False) == []


def test_dark_and_frozen_frames():
    m = HealthMonitor(HealthConfig(dark_s=60, frozen_s=120), clock=Clock())
    dark = np.zeros((72, 128, 3), np.uint8)
    for t in range(0, 70, 2):
        m.observe_frame("c1", float(t), dark)
    assert "dark_frame" in kinds(m) and "frozen_frame" not in kinds(m)
    m2 = HealthMonitor(HealthConfig(frozen_s=120), clock=Clock())
    still = noisy(1)
    for t in range(0, 130, 2):
        m2.observe_frame("c1", float(t), still)  # поток «завис» — кадр байт в байт
    assert kinds(m2) == ["frozen_frame"]
    m3 = HealthMonitor(HealthConfig(frozen_s=120), clock=Clock())
    for t in range(0, 130, 2):
        m3.observe_frame("c1", float(t), noisy(t))  # пустой зал, но шум матрицы есть
    assert kinds(m3) == []


def sc(ts, z, to):
    return Event(ts, z, "state_change", {"to": to})


def test_flapping_long_wait_short_sessions_no_staff():
    m = HealthMonitor(HealthConfig(flap_per_hour=12, shift_min_s=3600), clock=Clock())
    m.observe_frame("c1", 0.0, None)
    evs = [sc(10.0 * i, "T1", "SERVED" if i % 2 else "FREE") for i in range(20)]
    evs += [Event(100.0, "T2", "seated"), sc(100.0, "T2", "SEATED_WAITING")]
    for k in range(3):
        evs += [Event(200.0 + k * 100, "T3", "seated"), Event(230.0 + k * 100, "T3", "left", {"visit_duration_s": 30})]
    m.observe_events("c1", evs)
    m._cams["c1"].last_ts = 100.0 + 2401
    got = kinds(m, live=False)
    assert "flapping" in got and "long_wait" in got and "short_sessions" in got
    m.observe_frame("c1", 4000.0, None)
    assert "no_staff_visits" in kinds(m, live=False)  # 4 посадки за час работы и ни одного визита


def test_unknown_share():
    m = HealthMonitor(HealthConfig(unknown_min_obs=50, unknown_max=0.5), clock=Clock())
    for t in range(60):
        m.observe_roles("c1", float(t), ["unknown", "unknown", "guest"])
    assert "unknown_share" in kinds(m, live=False)
    m2 = HealthMonitor(HealthConfig(unknown_min_obs=50), clock=Clock())
    for t in range(60):
        m2.observe_roles("c1", float(t), ["guest", "staff", "unknown"])
    assert kinds(m2, live=False) == []


def test_check_events_offline():
    evs = [{"ts": 10.0 * i, "zone_id": "T1", "type": "state_change", "payload": {"to": "FREE"}, "camera_id": "c"}
           for i in range(30)]
    assert [i.kind for i in check_events(evs)] == ["flapping"]


def test_daily_helpers(tmp_path):
    assert due(time.strptime("2026-11-13 23:41", "%Y-%m-%d %H:%M"), "23:40", None)
    assert not due(time.strptime("2026-11-13 23:41", "%Y-%m-%d %H:%M"), "23:40", "2026-11-13")
    assert not due(time.strptime("2026-11-13 22:00", "%Y-%m-%d %H:%M"), "23:40", None)
    cap = caption_for("12:30", {"T1": {"state": "SEATED_WAITING", "wait_s": 240}, "T2": {"state": "FREE"}})
    assert cap == "12:30 · T1: ждут 4 мин, T2: свободен"
    txt = summary_text("Кафе", {"status": "warn", "issues": [{"camera_id": "c", "message": "кадр тёмный"}]},
                       [{"type": "seated"}, {"type": "seated"}, {"type": "served"}],
                       [{"confident": True}, {"confident": False}], [{"alert_id": "x"}])
    assert "Посадок: 2" in txt and "неуверенных (не отправлены администратору): 1" in txt and "кадр тёмный" in txt

    class R:
        camera_id, zone_states = "c", {"T1": {"state": "FREE"}}

        def __init__(self, ts):
            self.ts = ts

    s = FrameSampler(every_s=600)
    img = np.zeros((100, 200, 3), np.uint8)
    added = [s.maybe_add(R(float(t)), img, "10:00") for t in range(0, 7200, 60)]
    assert sum(added) == 12 and len(s.pick(5)) == 5


class FakeHTTP:
    def __init__(self, updates=None):
        self.calls = []
        self.updates = updates or []

    def post(self, url, json=None, data=None, files=None):
        self.calls.append((url.rsplit("/", 1)[-1], json, data, files))

        class Resp:
            status_code = 200
            text = "{}"

            def __init__(s, body):
                s._b = body

            def json(s):
                return s._b

        if url.endswith("getUpdates"):
            body, self.updates = {"ok": True, "result": self.updates}, []
            return Resp(body)
        return Resp({"ok": True})


def test_feedback_button_and_poller(tmp_path):
    http = FakeHTTP([{"update_id": 7, "callback_query": {"id": "q1", "data": "fb:abc123", "from": {"username": "adm"}}},
                     {"update_id": 8, "message": {"text": "привет"}}])
    tg = TelegramSender("TOKEN", "42", client=http)
    tg.send("алерт", feedback_id="abc123")
    assert http.calls[0][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "fb:abc123"
    store = EventStore(tmp_path / "e.sqlite")
    store.add_alert("abc123", "cam", "T1", "no_waiter", 200.0, "алерт", True, True, time.time())
    poller = FeedbackPoller(tg, store, timeout_s=0)
    assert poller.poll_once() == 1 and poller._offset == 9
    fb = store.feedback()
    assert fb[0]["alert_id"] == "abc123" and fb[0]["verdict"] == "wrong" and fb[0]["zone_id"] == "T1"
    assert any(c[0] == "answerCallbackQuery" for c in http.calls)
    tg.send_photos([(b"\xff\xd8jpeg", "кадр 1"), (b"\xff\xd8jpeg", "кадр 2")])
    name, _, data, files = http.calls[-1]
    assert name == "sendMediaGroup" and len(files) == 2 and "attach://p0" in data["media"]
    store.close()


# ---------------------------------------------------------------- W8: уверенные алерты
def test_is_confident_rules():
    e = lambda **p: Event(1.0, "T1", "alert", {"kind": "no_waiter", **p})  # noqa: E731
    assert is_confident(e(by_role=True, confidence=0.8), 0.6)
    assert not is_confident(e(by_role=True, confidence=0.4), 0.6)
    assert not is_confident(e(by_role=False, confidence=0.9), 0.6)  # гости не подтверждены ролью
    assert is_confident(e(by_role=True, confidence=None), 0.6)
    assert is_confident(Event(1.0, "D", "alert", {"kind": "desk_unattended"}), 0.6)  # clinic — без ролей гостей


class Sink:
    def __init__(self):
        self.sent = []

    def send(self, text, feedback_id=None):
        self.sent.append((text, feedback_id))


def test_dispatcher_holds_back_unconfident(tmp_path):
    store = EventStore(tmp_path / "e.sqlite")
    sink = Sink()
    d = AlertDispatcher(sink, "Кафе", min_confidence=0.6, store=store)
    ok = Event(200.0, "T1", "alert", {"kind": "no_waiter", "waited_s": 190, "by_role": True, "confidence": 0.8,
                                     "sid": 1}, "cam")
    weak = Event(210.0, "T2", "alert", {"kind": "no_waiter", "waited_s": 190, "by_role": False, "confidence": 0.3,
                                       "sid": 2}, "cam")
    recs = d.handle([ok, weak])
    d.close(drain_s=2)
    assert [r.confident for r in recs] == [True, False]
    assert len(sink.sent) == 1 and "Стол 1" in sink.sent[0][0] and sink.sent[0][1] == recs[0].alert_id
    assert d.held_back == 1 and len(d.feed) == 2  # неуверенный — в панели/отчёте
    rows = {r["zone_id"]: r for r in store.alerts_sent()}
    assert rows["T1"]["confident"] == 1 and rows["T1"]["sent"] == 1
    assert rows["T2"]["confident"] == 0 and rows["T2"]["sent"] == 0
    store.close()


def test_machine_alert_carries_confidence():
    """Сессия, где гости подтверждены ролью и уверенность ролей 0.9 → алерт уверенный; без роли — нет."""
    def run(conf, by_role_guests=True):
        m = TableMachine("T1", RestaurantThresholds(preexisting_grace_s=0), AlertThresholds(no_waiter_s=60))
        evs = []
        for i in range(int(120 * 5)):
            ts = i / 5
            g = (1, 2) if ts >= 10 else ()
            obs = ZoneObservation(ts=ts, zone_id="T1", guest_tracks=g if by_role_guests else (),
                                  unknown_tracks=() if by_role_guests else g,
                                  guest_confidence=conf if (g and by_role_guests) else None)
            evs += m.update(obs)
        return [e for e in evs if e.type == "alert"]

    a = run(0.9)
    assert a and a[0].payload["by_role"] and a[0].payload["confidence"] == 0.9 and is_confident(a[0], 0.6)
    b = run(0.4)
    assert b and not is_confident(b[0], 0.6)
    c = run(None, by_role_guests=False)  # сидят «неизвестные» (стол занят по присутствию)
    assert all(not is_confident(x, 0.6) for x in c)  # если алерт и есть — он не уходит в Telegram
