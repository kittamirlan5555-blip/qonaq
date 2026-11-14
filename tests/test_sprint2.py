"""SPRINT2: многокамерные потоки (S1) и прочее, что не завязано на живое железо."""
import os
import threading
import time

import numpy as np

from qonaq.config import CameraConfig, SiteConfig, expand_env
from qonaq.detect.base import FakeDetector
from qonaq.health.monitor import HealthMonitor
from qonaq.health.proc import CpuMeter, rss_mb
from qonaq.pipeline import Pipeline


class StreamStub:
    """Поток для теста: kind='stream', poll() отдаёт кадр, если `alive`, иначе None сразу."""

    kind = "stream"

    def __init__(self, alive: bool):
        self.alive = alive
        self.reconnects = 0
        self.connected = alive

    def open(self):
        return self

    def close(self):
        pass

    def poll(self):
        if not self.alive:
            return None
        return np.zeros((48, 64, 3), np.uint8), time.time()

    def read(self):  # блокирующее чтение мёртвого потока висело бы вечно — пайплайн не должен его звать
        raise AssertionError("read() у потока в многокамерном цикле")


def test_dead_stream_does_not_block_other_cameras():
    site = SiteConfig(cameras=[CameraConfig(id="a", source="rtsp://x/a"), CameraConfig(id="b", source="rtsp://x/b")])
    srcs = {"a": StreamStub(alive=False), "b": StreamStub(alive=True)}
    pipe = Pipeline(site, detector=FakeDetector(), sources=srcs)
    pipe.open()
    got = []
    for _ in range(5):
        res = pipe.step()
        assert res is not None  # потоки не «заканчиваются»
        got += [r.camera_id for r in res]
    assert got and set(got) == {"b"}
    st = pipe.stats.cams["b"]
    assert st.frames == len(got) and st.latency_max >= 0
    assert pipe.stats.cams["b"].window(time.time(), 10)["latency_ms_avg"] is not None
    srcs["a"].alive = False
    srcs["b"].alive = False
    assert pipe.step() == []  # оба молчат: пустой шаг, а не конец


def test_health_registers_camera_and_records_outage():
    now = {"t": 1000.0}
    mon = HealthMonitor(clock=lambda: now["t"])
    mon.register("cam1")
    now["t"] += 45  # ни одного кадра за 45 с — камера «не отвечает»
    kinds = [i.kind for i in mon.check(live=True)]
    assert "camera_down" in kinds
    frame = np.full((36, 64, 3), 100, np.uint8)
    mon.observe_frame("cam1", 0.0, frame)
    st = mon.status(live=True)
    assert st["cameras"]["cam1"]["outages"] == 1 and st["cameras"]["cam1"]["down_s_total"] == 45.0
    assert not [i for i in st["issues"] if i["kind"] == "camera_down"]


def test_expand_env_default():
    os.environ["QONAQ_T_SET"] = "x"
    os.environ.pop("QONAQ_T_UNSET", None)
    assert expand_env("${QONAQ_T_SET}/${QONAQ_T_UNSET:-def}/${QONAQ_T_UNSET}") == "x/def/${QONAQ_T_UNSET}"
    assert expand_env("${QONAQ_T_SET:-def}") == "x"


def test_proc_stats_work_here():
    rss, kind = rss_mb()
    assert rss is None or rss > 1
    m = CpuMeter()
    t = threading.Thread(target=lambda: sum(i * i for i in range(200000)))
    t.start()
    t.join()
    assert m.sample() >= 0


# ---------------------------------------------------------------- S4: Telegram на фейковом API
def test_telegram_full_flow_on_fake_api():
    from fastapi.testclient import TestClient

    from qonaq.alerts.cli_telegram import run_flow
    from qonaq.alerts.fake_api import FakeTelegramState, create_fake_app
    from qonaq.alerts.telegram import TelegramSender

    state = FakeTelegramState(token="T0K")
    client = TestClient(create_fake_app(state))
    checks = run_flow(lambda chat: TelegramSender("T0K", chat, client=client, base_url="http://testserver"), state)
    assert checks and not [c for c in checks if c.endswith("False")], checks
    chats = {m["chat_id"] for m in state.messages}
    assert chats == {"100", "200", "300"}
    admin = [m for m in state.messages if m["chat_id"] == "100"]
    assert admin[0]["reply_markup"]["inline_keyboard"][0][0]["text"] == "Неверно"
    assert "Стол 5" in admin[0]["text"] and "T05" not in admin[0]["text"]
    photos = [m for m in state.messages if m["photos"]]
    assert any("свободен" in m["text"] for m in photos)  # подписи кадров дошли в UTF-8
    # чужой токен — 401, как у настоящего API
    try:
        TelegramSender("wrong", "1", client=client, base_url="http://testserver").send("x")
        raise AssertionError("ожидалась ошибка")
    except RuntimeError as e:
        assert "401" in str(e)


def test_alert_expires_after_long_outage():
    from qonaq.alerts.telegram import AlertDispatcher
    from qonaq.logic.base import Event

    class Down:
        def send(self, text):
            raise ConnectionError("нет сети")

    clock = {"t": 1000.0}
    d = AlertDispatcher(Down(), retry_base_s=0.01, retry_max_s=0.02, max_age_s=600, clock=lambda: clock["t"])
    d.handle([Event(1.0, "T01", "alert", {"kind": "no_waiter", "sid": 1, "waited_s": 180.0}, "c")])
    deadline = time.time() + 3
    while d.failed_attempts < 2 and time.time() < deadline:
        time.sleep(0.01)
    clock["t"] += 3600
    while d.expired < 1 and time.time() < deadline:
        time.sleep(0.01)
    d.close()
    assert d.expired == 1 and d.sent_count == 0


def test_shift_summary_text_is_plain_russian():
    from qonaq.alerts.cli_telegram import example_messages

    ex = example_messages(wall=1_763_150_000.0)
    shift = ex["shift"]
    assert "итоги дня" in shift and "медиан" not in shift and "p90" not in shift
    assert "3 стола из 14" in shift
    for text in ex["alerts"].values():
        assert text.startswith("⚠ ") and "Стол " in text and len(text) < 120
