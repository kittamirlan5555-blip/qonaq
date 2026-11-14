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
