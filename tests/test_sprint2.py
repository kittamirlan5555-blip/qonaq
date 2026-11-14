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


# ---------------------------------------------------------------- S3: setup и служба
def test_rtsp_describe_digest_and_errors():
    """Мини-RTSP-сервер с Digest: 401 без авторизации, 200 с верным паролем, 404 на чужом пути."""
    import hashlib
    import re
    import socket

    from qonaq.setup.rtsp import describe

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    port = srv.getsockname()[1]
    realm, nonce = "IP Camera", "abc123"

    def handle(conn):
        with conn:
            while True:
                data = conn.recv(4096).decode()
                if not data:
                    return
                url = data.split()[1]
                cseq = re.search(r"CSeq: (\d+)", data).group(1)
                auth = re.search(r'response="([0-9a-f]+)"', data)
                if not auth:
                    conn.sendall(f'RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\nWWW-Authenticate: Digest '
                                 f'realm="{realm}", nonce="{nonce}"\r\n\r\n'.encode())
                    continue
                ha1 = hashlib.md5(f"admin:{realm}:s3cr3t".encode()).hexdigest()
                ha2 = hashlib.md5(f"DESCRIBE:{url}".encode()).hexdigest()
                good = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest() == auth.group(1)
                code = ("200 OK" if url.endswith("/Streaming/Channels/102") else "404 Not Found") if good else \
                    "401 Unauthorized"
                conn.sendall(f"RTSP/1.0 {code}\r\nCSeq: {cseq}\r\nServer: fake\r\n\r\n".encode())

    def serve():
        for _ in range(3):
            c, _ = srv.accept()
            handle(c)

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    base = f"rtsp://admin:{{}}@127.0.0.1:{port}"
    assert describe(base.format("s3cr3t") + "/Streaming/Channels/102").status == 200
    assert describe(base.format("wrong") + "/Streaming/Channels/102").status == 401
    assert describe(base.format("s3cr3t") + "/Streaming/Channels/902").status == 404
    srv.close()
    assert describe("rtsp://a:b@127.0.0.1:1/x").status == 0


def test_detect_vendor_stops_on_wrong_password():
    from qonaq.setup import discover
    from qonaq.setup.rtsp import RtspProbe

    calls = []

    def probe(url):
        calls.append(url)
        return RtspProbe(url, 401)

    orig = discover.tcp_open
    discover.tcp_open = lambda *a, **k: True
    try:
        res = discover.detect_vendor("10.0.0.5", "admin", "bad", probe=probe)
        assert res.vendor is None and "пароль" in res.error and len(calls) == 2  # не больше двух неудачных входов

        def probe_dahua(url):
            return RtspProbe(url, 200 if "realmonitor" in url and "channel=1&" in url or
                             ("realmonitor" in url and "channel=2&" in url) else 404)

        res = discover.detect_vendor("10.0.0.5", "admin", "ok", probe=probe_dahua)
        assert res.vendor == "dahua"
        discover.scan_channels(res, "admin", "ok", probe=probe_dahua)
        assert res.channels == [1, 2]
    finally:
        discover.tcp_open = orig


def test_build_url_encodes_only_breaking_chars():
    from qonaq.setup.discover import build_url

    assert build_url("hikvision", "1.2.3.4", "admin", "Pa$$!", 3) == \
        "rtsp://admin:Pa$$!@1.2.3.4:554/Streaming/Channels/302"
    assert "p%40ss" in build_url("dahua", "1.2.3.4", "admin", "p@ss", 1)


def test_onvif_probe_match_parsing():
    from qonaq.setup.discover import parse_probe_match

    xml = ('<s:Envelope><s:Body><d:ProbeMatches><d:ProbeMatch><d:Scopes>onvif://www.onvif.org/hardware/DS-7608NI '
           'onvif://www.onvif.org/name/HIKVISION</d:Scopes><d:XAddrs>http://192.168.1.64/onvif/device_service '
           'http://[fe80::1]/onvif/device_service</d:XAddrs></d:ProbeMatch></d:ProbeMatches></s:Body></s:Envelope>')
    out = parse_probe_match(xml)
    assert out[0]["ip"] == "192.168.1.64" and out[0]["hardware"] == "DS-7608NI" and out[0]["name"] == "HIKVISION"


def test_update_env_keeps_other_lines(tmp_path):
    from qonaq.setup.wizard import update_env

    env = tmp_path / ".env"
    env.write_text("TELEGRAM_BOT_TOKEN=x\nQONAQ_NVR_PASSWORD=old\n", encoding="utf-8")
    update_env(env, {"QONAQ_NVR_PASSWORD": "new", "QONAQ_NVR_USER": "admin"})
    assert env.read_text(encoding="utf-8").splitlines() == ["TELEGRAM_BOT_TOKEN=x", "QONAQ_NVR_PASSWORD=new",
                                                            "QONAQ_NVR_USER=admin"]


def test_service_files_are_generated():
    from pathlib import Path

    from qonaq.setup.service import service_name, systemd_unit, windows_wrapper

    assert service_name("cafe_arman") == "qonaq-cafe-arman"
    unit = systemd_unit("cafe", "configs/sites/cafe.yaml", "/opt/q/.venv/bin/python", Path("/opt/q"), False, "qonaq")
    assert "Restart=always" in unit and "User=qonaq" in unit and "WantedBy=multi-user.target" in unit
    assert "ExecStart=/opt/q/.venv/bin/python -m qonaq live --site configs/sites/cafe.yaml" in unit
    user_unit = systemd_unit("cafe", "configs/sites/cafe.yaml", "/p", Path("/opt/q"), True, None)
    assert "User=" not in user_unit and "WantedBy=default.target" in user_unit
    cmd = windows_wrapper("cafe", "configs/sites/cafe.yaml", r"C:\Qonaq\.venv\Scripts\python.exe", Path("C:/Qonaq"))
    assert ":loop" in cmd and "goto loop" in cmd and "timeout" not in cmd.split("ping")[1]


# ---------------------------------------------------------------- S5: журнал, периоды, отчёт пилота
def test_sessions_survive_restart_with_same_sid(tmp_path):
    """После перезапуска службы sid начинаются заново — старые сессии не должны затираться."""
    import sqlite3

    from qonaq.events.store import EventStore
    from qonaq.logic.base import Event

    db = tmp_path / "events.sqlite"
    s = EventStore(db)
    s.add([Event(100.0, "T01", "seated", {"sid": 1}, "c"), Event(200.0, "T01", "left", {"sid": 1}, "c")])
    s.close()
    s = EventStore(db)  # «перезапуск»
    s.add([Event(5000.0, "T01", "seated", {"sid": 1}, "c")])
    assert len(s.sessions()) == 2
    s.close()
    # миграция журнала со старой схемой (UNIQUE без seated_ts)
    old = tmp_path / "old.sqlite"
    c = sqlite3.connect(old)
    c.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, zone_id TEXT "
              "NOT NULL, sid INTEGER NOT NULL, kind TEXT NOT NULL DEFAULT 'table', seated_ts REAL NOT NULL, served_ts "
              "REAL, left_ts REAL, cleaned_ts REAL, response_time_s REAL, visit_duration_s REAL, cleanup_time_s REAL, "
              "staff_visits INTEGER NOT NULL DEFAULT 0, guests INTEGER, outcome TEXT NOT NULL DEFAULT 'open', "
              "preexisting INTEGER NOT NULL DEFAULT 0, dirty_seat INTEGER NOT NULL DEFAULT 0, low_confidence INTEGER "
              "NOT NULL DEFAULT 0, UNIQUE(camera_id, zone_id, sid, kind))")
    c.execute("INSERT INTO sessions(camera_id, zone_id, sid, seated_ts) VALUES ('c', 'T01', 1, 10.0)")
    c.commit()
    c.close()
    s = EventStore(old)
    s.add([Event(900.0, "T01", "seated", {"sid": 1}, "c")])
    assert len(s.sessions()) == 2
    s.close()


def test_slice_and_merge_periods():
    from qonaq.events.store import EventStore
    from qonaq.logic.base import Event
    from qonaq.metrics.period import slice_store

    s = EventStore(":memory:")
    s.add([Event(t, "T01", typ, {"sid": i, "response_time_s": 60.0}, "c") for i, t0 in enumerate((1000, 90000))
           for t, typ in ((t0, "seated"), (t0 + 60, "served"))])
    s.set_meta(time_base="epoch", tables=1)
    day2 = slice_store(s, 86400, 2 * 86400)
    assert [x["seated_ts"] for x in day2.sessions()] == [90000]
    assert day2.meta()["t_start"] == 86400


def test_synth_week_and_pilot_report(tmp_path):
    from qonaq.config import SiteConfig
    from qonaq.metrics.pilot import generate_pilot_report
    from qonaq.synth.week import WeekParams, simulate

    site = SiteConfig(name="Тестовое кафе")
    simulate(WeekParams(name="b", days=2, seed=1, response_median_s=200, cleanup_median_s=300), site,
             tmp_path / "b").close()
    simulate(WeekParams(name="a", days=2, seed=2, start="2026-11-09", response_median_s=90, cleanup_median_s=100),
             site, tmp_path / "a").close()
    out, A, B = generate_pilot_report(site, [str(tmp_path / "b")], [str(tmp_path / "a")], tmp_path / "r.html")
    assert A["days"] == 2 and B["days"] == 2
    assert A["seatings_day"] > 30  # сессии не затёрлись между днями
    assert B["median"] < A["median"]
    text = out.read_text(encoding="utf-8")
    assert "СИНТЕТИЧЕСКИХ" in text and "Как было и как стало" in text and "<svg" in text and "медиан" in text


# ---------------------------------------------------------------- S2: плитки, ROI, детектор через кадр
def test_tile_boxes_cover_roi():
    from qonaq.detect.onnx_yolox import tile_boxes

    assert tile_boxes(1920, 1080, 1) == []
    assert tile_boxes(1920, 1080, 2) == [(0, 0, 1152, 1080), (768, 0, 1920, 1080)]  # прежнее поведение
    roi = (900, 200, 1920, 800)
    boxes = tile_boxes(1920, 1080, 4, roi)
    assert 4 <= len(boxes) <= 6  # область 1020×600: около 4 плиток по пропорции
    assert min(b[0] for b in boxes) == 900 and max(b[2] for b in boxes) <= 1920
    assert min(b[1] for b in boxes) == 200 and max(b[3] for b in boxes) <= 800


def test_detect_every_reuses_last_detections():
    from qonaq.detect.base import Detection, FakeDetector

    class Counting(FakeDetector):
        def detect(self, frames):
            self.calls += 1
            return [[Detection(10, 10, 20, 40, 0.9, "person")] for _ in frames]

    site = SiteConfig(cameras=[CameraConfig(id="a", source="rtsp://x/a")])
    site.detector.detect_every = 3
    det = Counting()
    pipe = Pipeline(site, detector=det, sources={"a": StreamStub(alive=True)})
    pipe.open()
    n = 0
    while n < 9:
        n += len(pipe.step())
        time.sleep(0.001)
    assert det.calls == 3
