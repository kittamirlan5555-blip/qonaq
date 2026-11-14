"""B8: перезапуск службы / потеря питания посреди сессии стола или игры."""
from __future__ import annotations

from qonaq.billiard.evaluate import sessions_from_events
from qonaq.config import SiteConfig
from qonaq.events.store import EventStore
from qonaq.logic.base import Event
from qonaq.metrics.compute import compute_metrics
from qonaq.metrics.period import slice_store

T = 1_800_000_000.0


def _hb(ts):
    return Event(ts, "", "heartbeat", {"window_s": 60, "up": ["cam"]}, "")


def _restart_journal(path, guests_still_there: bool):
    st = EventStore(path)
    st.set_meta(time_base="epoch", t_start=T)
    st.add([Event(T + 60, "T01", "seated", {"sid": 1}, "cam")] + [_hb(T + 60 * i) for i in range(1, 11)])
    st.close()
    st = EventStore(path)  # новый процесс: в памяти ничего нет
    closed = st.close_interrupted()
    assert [(e.type, e.zone_id, e.ts) for e in closed] == [("aborted", "T01", T + 600)]
    if guests_still_there:  # движок после старта видит стол занятым — «уже сидели»
        st.add([Event(T + 640, "T01", "seated", {"sid": 1, "preexisting": True}, "cam"),
                Event(T + 1800, "T01", "left", {"visit_duration_s": 1160}, "cam")])
    st.add([_hb(T + 1860)])
    return st


def test_restart_mid_session_continues_same_session(tmp_path):
    st = _restart_journal(tmp_path / "e.sqlite", guests_still_there=True)
    ss = st.sessions()
    assert len(ss) == 1 and ss[0]["seated_ts"] == T + 60 and ss[0]["outcome"] == "left"
    assert ss[0]["left_ts"] == T + 1800 and not ss[0]["preexisting"]
    m = compute_metrics(st, SiteConfig())
    assert m["turnover"]["seatings"] == 1
    # срез (как в сводке смены) переигрывает события — результат тот же
    sl = slice_store(st, T, T + 2000)
    s2 = sl.sessions()
    assert len(s2) == 1 and s2[0]["outcome"] == "left" and s2[0]["seated_ts"] == T + 60


def test_restart_table_empty_afterwards_is_closed_not_open_forever(tmp_path):
    st = _restart_journal(tmp_path / "e.sqlite", guests_still_there=False)
    ss = st.sessions()
    assert len(ss) == 1 and ss[0]["outcome"] == "aborted" and ss[0]["left_ts"] == T + 600
    m = compute_metrics(st, SiteConfig())
    assert m["response"]["unserved_over_threshold"] == 0  # не «ждут до сих пор»
    # повторный старт ничего не закрывает второй раз
    assert st.close_interrupted() == []


def test_restart_long_outage_starts_new_session(tmp_path):
    st = EventStore(tmp_path / "e.sqlite")
    st.add([Event(T + 60, "T01", "seated", {"sid": 1}, "cam"), _hb(T + 120)])
    st.close()
    st = EventStore(tmp_path / "e.sqlite")
    st.close_interrupted()
    st.add([Event(T + 120 + 3600, "T01", "seated", {"sid": 1, "preexisting": True}, "cam")])  # через час — другие
    assert [s["outcome"] for s in st.sessions()] == ["aborted", "open"]


def test_billiard_game_interrupted_by_power_loss_ends_at_last_confirmation(tmp_path):
    st = EventStore(tmp_path / "e.sqlite")
    st.add([Event(T + 100, "B1", "game_started", {"start": T + 100, "detected_ts": T + 130}, "cam"),
            Event(T + 900, "B1", "state_change", {"from": "IN_PLAY", "to": "PAUSED"}, "cam"),
            Event(T + 2000, "B2", "game_started", {"start": T + 2000, "detected_ts": T + 2030}, "cam"),
            Event(T + 2100, "B2", "game_ended", {"start": T + 2000, "end": T + 2100}, "cam"),
            _hb(T + 5000)])
    st.close()
    st = EventStore(tmp_path / "e.sqlite")
    closed = st.close_interrupted()
    assert [(e.zone_id, e.type, e.payload["end"], e.payload["reason"]) for e in closed] == [
        ("B1", "game_ended", T + 900, "restart")]
    ses = sessions_from_events(st.events())
    assert ses["B1"] == [(T + 100, T + 900)]  # не до последнего события другого стола (T+5000)


def test_billiard_open_game_overwritten_by_new_start_is_kept():
    evs = [{"ts": T, "type": "game_started", "zone_id": "B1", "camera_id": "c", "payload": {"start": T,
                                                                                          "detected_ts": T + 30}},
           {"ts": T + 900, "type": "game_started", "zone_id": "B1", "camera_id": "c",
            "payload": {"start": T + 900, "detected_ts": T + 930}},
           {"ts": T + 1500, "type": "game_ended", "zone_id": "B1", "camera_id": "c",
            "payload": {"start": T + 900, "end": T + 1500}}]
    assert sessions_from_events(evs)["B1"] == [(T, T + 30), (T + 900, T + 1500)]


def test_steady_clock_survives_wall_clock_jumps():
    """B8: часы ПК откатились на час — время событий не идёт назад и кадры не перестают обрабатываться."""
    from qonaq.sources.video import SteadyClock, VideoSource

    w = {"t": 1_000_000.0}
    m = {"t": 50.0}
    c = SteadyClock(wall=lambda: w["t"], mono=lambda: m["t"])
    seen = []
    for _ in range(10):
        m["t"] += 0.5
        w["t"] += 0.5
        seen.append(c())
    w["t"] -= 3600  # NTP откатил часы на час
    for _ in range(10):
        m["t"] += 0.5
        w["t"] += 0.5
        seen.append(c())
    assert all(b > a for a, b in zip(seen, seen[1:], strict=False))
    assert abs((seen[-1] - seen[0]) - 9.5) < 1e-6
    w["t"] += 3600 + 7200  # а теперь вперёд на 2 ч: переякориться (это разрыв, как обрыв камеры)
    m["t"] += 0.5
    assert abs(c() - w["t"]) < 1e-6 and c.jumps == 1

    # VideoSource.poll с откатившимися часами продолжает отдавать кадры с частотой process_fps
    src = VideoSource.__new__(VideoSource)
    src.kind, src.process_fps, src._last_emit = "stream", 2.0, 0.0
    src._clock = c
    import threading

    src._lock = threading.Lock()
    got = 0
    for _ in range(20):
        m["t"] += 0.5
        w["t"] -= 100  # часы ПК всё время уходят назад
        src._latest = ("frame", c())
        got += src.poll() is not None
    assert got >= 19


def test_shift_and_daily_summary_not_resent_after_restart(tmp_path):
    from qonaq.api.service import LiveService
    from qonaq.config import AlertThresholds, CameraConfig, HealthSettings

    class Sink:
        def __init__(self):
            self.texts = []

        def send(self, text, **_):
            self.texts.append(text)

        def send_photos(self, *_a, **_k):
            pass

    site = SiteConfig(site_id="t", name="Тест", runs_dir=str(tmp_path),
                      cameras=[CameraConfig(id="cam", source=str(tmp_path / "none.mp4"))],
                      alerts=AlertThresholds(shift_summary_at="00:00"), health=HealthSettings(daily_at="00:00"))
    sent = []
    for _ in range(2):  # два запуска службы в один день
        svc = LiveService(site, use_env_sender=False, run_dir=tmp_path / "live")
        svc.venue, svc.owner = Sink(), Sink()
        svc.health_tick()
        sent.append((len(svc.venue.texts), len([t for t in svc.owner.texts if "сводка за день" in t])))
        svc.stop()
    assert sent == [(1, 1), (0, 0)]


def test_health_monitor_status_while_pipeline_writes():
    """B8: status() из потока здоровья/панели, пока пайплайн пишет наблюдения, — без «mutated during iteration»."""
    import threading
    import time

    from qonaq.health.monitor import HealthMonitor

    mon = HealthMonitor()
    mon.register("cam")
    stop = threading.Event()

    def writer():
        t = 0.0
        while not stop.is_set():
            t += 0.1
            mon.observe_roles("cam", t, ["guest", "unknown", "staff"] * 5)
            mon.observe_events("cam", [Event(t, "T01", "state_change", {"from": "FREE", "to": "SEATED_WAITING"}),
                                       Event(t, "T01", "seated", {"sid": 1})])

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    errors = 0
    t_end = time.time() + 1.5
    while time.time() < t_end:
        try:
            mon.status(live=False)
        except RuntimeError:
            errors += 1
    stop.set()
    th.join(2)
    assert errors == 0


def test_timezone_fallback_is_the_same_everywhere(monkeypatch):
    """B8: без базы поясов (Windows без tzdata) алерт, отчёт и сверка показывают одно и то же время."""
    import zoneinfo

    from qonaq.metrics.compute import fmt_ts
    from qonaq.reconcile import local_tz
    from qonaq.strings import clock

    def boom(*_a, **_k):
        raise zoneinfo.ZoneInfoNotFoundError("нет tzdata")

    monkeypatch.setattr(zoneinfo, "ZoneInfo", boom)
    monkeypatch.setenv("TZ", "Etc/GMT-6")  # пояс ПК — +6 (устаревший пояс РК)
    ts = 1_794_690_000.0
    from datetime import datetime

    want = datetime.fromtimestamp(ts, local_tz("Asia/Almaty")).strftime("%H:%M")
    assert clock(ts, "Asia/Almaty") == want == fmt_ts(ts, True)
    assert local_tz("Asia/Almaty").utcoffset(None).total_seconds() == 5 * 3600
