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
