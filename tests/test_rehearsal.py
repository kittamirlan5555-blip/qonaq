"""B7: находки репетиции установки (docs/REHEARSAL_WINDOWS.md)."""
from __future__ import annotations

import logging

from qonaq import config
from qonaq.config import load_dotenv_if_any, unresolved_env
from qonaq.sources.video import parse_source


def test_dotenv_found_in_project_root_when_cwd_elsewhere(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".env").write_text("QONAQ_TEST_REH_USER=admin\nQONAQ_TEST_REH_PW=from_root\n", encoding="utf-8")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.delenv("QONAQ_TEST_REH_USER", raising=False)
    monkeypatch.delenv("QONAQ_TEST_REH_PW", raising=False)
    monkeypatch.chdir(other)
    loaded = load_dotenv_if_any(other)
    assert [p.parent for p in loaded] == [root]
    import os

    assert os.environ["QONAQ_TEST_REH_PW"] == "from_root"


def test_dotenv_cwd_wins_over_project_root(tmp_path, monkeypatch):
    root, cwd = tmp_path / "proj", tmp_path / "cwd"
    root.mkdir()
    cwd.mkdir()
    (root / ".env").write_text("QONAQ_TEST_REH_PW=root\n", encoding="utf-8")
    (cwd / ".env").write_text("QONAQ_TEST_REH_PW=cwd\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.delenv("QONAQ_TEST_REH_PW", raising=False)
    assert len(load_dotenv_if_any(cwd)) == 2
    import os

    assert os.environ["QONAQ_TEST_REH_PW"] == "cwd"


def test_same_dir_loaded_once(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("QONAQ_TEST_REH_X=1\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    assert len(load_dotenv_if_any(tmp_path)) == 1


def test_unresolved_env_in_camera_url_is_reported(monkeypatch, caplog):
    monkeypatch.delenv("QONAQ_TEST_REH_NOPE", raising=False)
    monkeypatch.setenv("QONAQ_TEST_REH_YES", "x")
    url = "rtsp://${QONAQ_TEST_REH_NOPE}:${QONAQ_TEST_REH_YES}@1.2.3.4:554/${QONAQ_TEST_REH_D:-ok}"
    assert unresolved_env(url) == ["QONAQ_TEST_REH_NOPE"]
    with caplog.at_level(logging.ERROR, logger="qonaq.source"):
        kind, target = parse_source(url)
    assert kind == "stream" and "${QONAQ_TEST_REH_NOPE}" in target
    assert any("QONAQ_TEST_REH_NOPE" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- простои по heartbeat (переживают перезапуск)
def _hb(ts, up, window=60.0, final=False):
    p = {"window_s": window, "up": up}
    if final:
        p["final"] = True
    return {"ts": ts, "type": "heartbeat", "zone_id": "", "camera_id": "", "payload": p}


def test_downtime_crash_clean_stop_and_camera_outage():
    from qonaq.health.downtime import downtime, downtime_lines

    t = 1_000_000.0
    evs = [_hb(t + 60 * i, ["a", "b"]) for i in range(1, 4)]          # 0..180 всё работает
    evs += [_hb(t + 240, ["a"]), _hb(t + 300, ["a"])]                  # b нет 2 окна: 180..300
    evs += [_hb(t + 360, ["a", "b"])]
    # аварийный разрыв: следующий heartbeat через 10 мин, его окно — с момента нового старта
    evs += [_hb(t + 960, ["a", "b"], window=60.0)]
    # штатная остановка и запуск через 5 мин
    evs += [_hb(t + 990, ["a", "b"], window=30.0, final=True), _hb(t + 1350, ["a", "b"])]
    evs += [{"ts": t + 100, "type": "seated", "zone_id": "T1", "camera_id": "a", "payload": {}}]
    gaps = downtime(evs)
    kinds = [(g.camera_id, round(g.start - t), round(g.end - t), g.clean) for g in gaps]
    assert kinds == [("b", 180, 300, None), ("", 360, 900, False), ("", 990, 1290, True)]
    lines = downtime_lines(gaps)
    assert "аварийно" in lines[1] and "штатно" in lines[2]


def test_downtime_final_heartbeat_is_not_camera_loss_and_no_gap_when_continuous():
    from qonaq.health.downtime import downtime

    t = 5_000.0
    evs = [_hb(t + 60 * i, ["a"]) for i in range(1, 30)] + [_hb(t + 60 * 29 + 20, [], window=20, final=True)]
    assert downtime(evs) == []
    assert downtime([]) == []


def test_daily_summary_counts_only_last_24h(tmp_path):
    import time

    from qonaq.events.store import EventStore
    from qonaq.health.daily import build_daily
    from qonaq.logic.base import Event

    now = time.time()
    st = EventStore(tmp_path / "events.sqlite")
    st.add([Event(now - 3 * 86400 + i, f"T{i}", "seated", {}, "cam") for i in range(5)])  # прошлые дни
    st.add([Event(now - 3600, "T1", "seated", {}, "cam"), Event(now - 1800, "", "heartbeat",
                                                                      {"window_s": 60, "up": ["cam"]}, "")])
    st.close()
    d = build_daily(tmp_path, "X", {"status": "ok", "issues": [], "cameras": {}}, [], since=now - 86400)
    assert "Посадок: 1," in d["text"]
    d_all = build_daily(tmp_path, "X", {"status": "ok", "issues": [], "cameras": {}}, [])
    assert "Посадок: 6," in d_all["text"]


def test_graceful_server_does_not_reraise_sigterm():
    import signal

    import uvicorn
    from fastapi import FastAPI

    from qonaq.api.app import graceful_server

    srv = graceful_server(uvicorn.Config(FastAPI(), host="127.0.0.1", port=0))
    before = signal.getsignal(signal.SIGTERM)
    with srv.capture_signals():
        srv.handle_exit(signal.SIGTERM, None)  # как если бы пришёл SIGTERM от systemd
        assert srv.should_exit
    # дошли сюда — процесс не убит повторным сигналом; обработчик восстановлен
    assert signal.getsignal(signal.SIGTERM) == before
