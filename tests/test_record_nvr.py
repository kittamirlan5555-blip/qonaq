"""scripts/record_nvr.py: каналы, основной поток, окно расписания, запись кусками (на файле вместо RTSP)."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("record_nvr", ROOT / "scripts" / "record_nvr.py")
rn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rn)


def test_parse_channels():
    assert rn.parse_channels("1-4,7") == [1, 2, 3, 4, 7]
    assert rn.parse_channels("3, 1,3") == [1, 3]


def test_main_stream_per_vendor():
    from qonaq.setup.discover import build_url

    h = build_url("hikvision", "10.0.0.2", "u", "p", 12)
    assert rn.main_stream("hikvision", h).endswith("/Streaming/Channels/1201")
    assert "subtype=0" in rn.main_stream("dahua", build_url("dahua", "10.0.0.2", "u", "p", 3))
    assert "/s0/live" in rn.main_stream("uniview", build_url("uniview", "10.0.0.2", "u", "p", 3))


def test_window_over_midnight_current_and_next():
    win = rn.parse_window("18:00-02:00")
    s, e = rn.window_at(dt.datetime(2026, 11, 14, 1, 0), win)  # ночью — идёт вчерашнее окно
    assert (s, e) == (dt.datetime(2026, 11, 13, 18, 0), dt.datetime(2026, 11, 14, 2, 0))
    s, e = rn.window_at(dt.datetime(2026, 11, 14, 12, 0), win)  # днём — ближайшее вечером
    assert (s, e) == (dt.datetime(2026, 11, 14, 18, 0), dt.datetime(2026, 11, 15, 2, 0))


def test_password_never_in_cmd_log(caplog):
    from qonaq.config import mask_secrets

    url = "rtsp://admin:S3cret!@10.0.0.2:554/Streaming/Channels/102"
    assert "S3cret" not in mask_secrets(url)


def test_record_file_into_segments(tmp_path):
    """Без регистратора: файл как поток (-re), 4 с кусками по 2 с; рядом json с началом записи."""
    src = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(80):
        f = np.full((120, 160, 3), (i * 3) % 255, np.uint8)
        w.write(f)
    w.release()
    try:
        rn.ffmpeg_exe()
    except SystemExit:
        pytest.skip("нет ffmpeg")
    end = dt.datetime.now() + dt.timedelta(seconds=4)
    rc = rn.record({"ch01": str(src)}, end, tmp_path / "nvr", segment_s=2, min_free_gb=0.0)
    assert rc == 0
    mp4 = sorted((tmp_path / "nvr").rglob("ch01_*.mp4"))
    assert len(mp4) >= 1 and all(p.stat().st_size > 0 for p in mp4)
    meta = json.loads(next((tmp_path / "nvr").rglob("ch01_*.json")).read_text(encoding="utf-8"))
    assert meta["channel"] == "ch01" and meta["started_epoch"] > 0


def test_stop_midway_keeps_last_segment_readable(tmp_path):
    """Ctrl+C / нехватка места посреди куска: ffmpeg получает «q» и дописывает mp4 — кусок открывается."""
    import threading
    import time

    src = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(300):
        w.write(np.full((120, 160, 3), (i * 3) % 255, np.uint8))
    w.release()
    try:
        rn.ffmpeg_exe()
    except SystemExit:
        pytest.skip("нет ffmpeg")
    stop = threading.Event()
    ch = rn.Channel("ch02", str(src), tmp_path / "nvr", dt.datetime.now() + dt.timedelta(seconds=60), 600, stop)
    ch.start()
    time.sleep(4)
    stop.set()
    ch.join(20)
    assert not ch.is_alive()
    mp4 = sorted((tmp_path / "nvr").rglob("ch02_*.mp4"))
    assert mp4
    cap = cv2.VideoCapture(str(mp4[-1]))
    assert cap.read()[0] and cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 10
