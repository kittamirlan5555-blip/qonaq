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


# ---------------------------------------------------------------- B1: «поток завис» против «никто не двигается»
def _scene(seed=7):
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), 110, np.uint8)
    for _ in range(12):  # «мебель» — неподвижные прямоугольники
        x, y = int(rng.integers(0, 560)), int(rng.integers(0, 300))
        img[y:y + 50, x:x + 70] = rng.integers(40, 200, 3)
    return img


def _sensor(img, sigma, seed):
    n = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)


def _osd(img, t):
    import cv2
    img = img.copy()
    cv2.putText(img, f"23:{int(t) // 60:02d}:{int(t) % 60:02d}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def test_still_live_scene_is_not_frozen():
    """Пустой зал: в кадре ничего не движется, но шум матрицы (σ=1) есть → не тревога, а still_s."""
    m = HealthMonitor(HealthConfig(frozen_s=120), clock=Clock())
    base = _scene()
    for t in range(0, 200, 2):
        m.observe_frame("c1", float(t), _sensor(base, 1.0, t))
    assert kinds(m) == []
    assert m.status()["cameras"]["c1"]["still_s"] >= 190


def test_frozen_stream_identical_frames():
    m = HealthMonitor(HealthConfig(frozen_s=120), clock=Clock())
    still = _sensor(_scene(), 2.0, 0)
    for t in range(0, 130, 2):
        m.observe_frame("c1", float(t), still)
    iss = m.check()
    assert [i.kind for i in iss] == ["frozen_frame"] and iss[0].data["osd"] is False
    assert "поток завис" in iss[0].message


def test_frozen_camera_with_ticking_osd_clock():
    """Регистратор рисует часы поверх застывшей картинки: меняется 1–3 клетки из 144 → всё равно завис."""
    m = HealthMonitor(HealthConfig(frozen_s=120), clock=Clock())
    still = _sensor(_scene(), 2.0, 0)
    for t in range(0, 130, 2):
        m.observe_frame("c1", float(t), _osd(still, t))
    iss = m.check()
    assert [i.kind for i in iss] == ["frozen_frame"] and iss[0].data["osd"] is True
    assert "время на картинке идёт" in iss[0].message


def test_frozen_clears_when_noise_returns_and_motion_resets_still():
    m = HealthMonitor(HealthConfig(frozen_s=60), clock=Clock())
    base = _scene()
    still = _sensor(base, 2.0, 0)
    for t in range(0, 70, 2):
        m.observe_frame("c1", float(t), still)
    assert kinds(m) == ["frozen_frame"]
    m.observe_frame("c1", 72.0, _sensor(base, 2.0, 72))
    assert kinds(m) == []
    moved = base.copy()
    moved[100:250, 200:300] = 30  # человек вошёл
    m.observe_frame("c1", 74.0, _sensor(moved, 2.0, 74))
    assert m.status()["cameras"]["c1"]["still_s"] == 0.0


def test_live_cells_fraction_basic():
    from qonaq.health.monitor import live_cells_fraction
    a = np.full((90, 160), 100, np.uint8)
    assert live_cells_fraction(a, a) == 0.0
    b = a.copy()
    b[:10, :10] = 120  # одна клетка
    assert 0 < live_cells_fraction(b, a) <= 2 / 144
    c = _sensor(a[..., None].repeat(3, 2), 1.0, 1)[..., 0]
    assert live_cells_fraction(c, a) > 0.9


def _encode_and_watch(tmp_path, name, frame_fn, seconds=130, fps=2.0):
    """Кадры → H.264 (как пишет VideoSink/регистратор) → декодирование → монитор."""
    import cv2

    from qonaq.render.sink import VideoSink
    p = tmp_path / f"{name}.mp4"
    sink = VideoSink(p, fps, crf=28)
    for k in range(int(seconds * fps)):
        sink.write(frame_fn(k))
    sink.close()
    m = HealthMonitor(HealthConfig(frozen_s=90), clock=Clock())
    cap = cv2.VideoCapture(str(p))
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        m.observe_frame("c1", k / fps, f)
        k += 1
    assert k > 0.9 * seconds * fps
    return m


def test_frozen_vs_still_through_h264(tmp_path):
    """Сквозь кодек: шум матрицы в P-кадрах почти съедается, но I-кадры живой камеры каждый раз новые,
    а зависшая картинка, пережатая кодеком, повторяется с периодом GOP."""
    from qonaq.synth.render import add_sensor_noise
    base = cv2_resize(_scene(), (320, 180))
    rng = np.random.default_rng(0)
    live = _encode_and_watch(tmp_path, "live", lambda k: add_sensor_noise(base, 1.5, rng))
    assert kinds(live) == [] and live.status()["cameras"]["c1"]["still_s"] > 100
    frozen = _encode_and_watch(tmp_path, "frozen", lambda k: base)
    assert kinds(frozen) == ["frozen_frame"]
    frozen_osd = _encode_and_watch(tmp_path, "osd", lambda k: _osd(base, k / 2))
    iss = frozen_osd.check()
    assert [i.kind for i in iss] == ["frozen_frame"] and iss[0].data["osd"] is True


def cv2_resize(img, size):
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
