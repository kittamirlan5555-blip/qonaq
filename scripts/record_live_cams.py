"""Запись публичных круглосуточных трансляций баров/ресторанов для ВНУТРЕННЕГО тестирования (docs/DATA_SOURCES.md).

Каждая камера пишется 1 час в вечернее время заведения (когда есть гости). Видео — только в data/videos/live/
(в .gitignore): не коммитить, не публиковать; кадры в отчётах — с блюром (render/blur).

    python scripts/record_live_cams.py               # по расписанию ниже (ждёт начала)
    python scripts/record_live_cams.py --now --minutes 2   # проверка: сразу, по 2 минуты

Нужны yt-dlp (`pip install yt-dlp`, путь — --ytdlp), JS-рантайм для YouTube (node) и ffmpeg в PATH.
Компьютер не уходит в сон, пока идёт запись или ожидание (SetThreadExecutionState на Windows — как у видеоплеера;
настройки системы не меняются).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "videos" / "live"

# (id, имя файла, часовой пояс UTC±ч, местное время начала) — 19:00–20:00 по местному: ужин, гости есть.
CAMS = [
    ("bM0GplM_vPY", "elbo_room_patio_fort_lauderdale", -5, "19:00"),
    ("wVNt3l657X0", "elbo_room_bar_fort_lauderdale", -5, "19:00"),
    ("thzfsD7VDFg", "hogs_breath_outside_key_west", -5, "19:00"),
    ("S605ycm0Vlk", "hogs_breath_minibar_key_west", -5, "19:00"),
    ("2wqpy036z24", "beach_bar_st_john_usvi", -4, "19:00"),
    ("d-drfinzIWU", "windmill_bar_st_john_usvi", -4, "19:00"),
]


def keep_awake(on: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def next_start_utc(tz_h: int, hhmm: str, now: dt.datetime) -> dt.datetime:
    h, m = map(int, hhmm.split(":"))
    local_now = now + dt.timedelta(hours=tz_h)
    start_local = local_now.replace(hour=h, minute=m, second=0, microsecond=0)
    if start_local + dt.timedelta(minutes=50) < local_now:  # сегодняшний вечер прошёл — завтра
        start_local += dt.timedelta(days=1)
    return start_local - dt.timedelta(hours=tz_h)


def log(msg: str) -> None:
    line = f"{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M:%S}Z {msg}"
    print(line, flush=True)
    with (OUT / "record.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def stream_url(ytdlp: str, vid: str, height: int) -> str | None:
    cmd = [ytdlp, "--js-runtimes", "node", "-g", "-f", f"bv*[height<={height}]/b", f"https://www.youtube.com/watch?v={vid}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    url = r.stdout.strip().splitlines()[0] if r.stdout.strip() else None
    if not url:
        log(f"{vid}: нет ссылки на поток: {r.stderr.strip()[-300:]}")
    return url


def record(ytdlp: str, vid: str, name: str, start: dt.datetime | None, seconds: float, height: int) -> None:
    if start is not None:
        wait = (start - dt.datetime.now(dt.UTC)).total_seconds()
        log(f"{name}: старт {start:%H:%M}Z (через {wait / 60:.0f} мин)")
        if wait > 0:
            time.sleep(wait)
    t_end = time.time() + seconds
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%MZ")
    part = 0
    while time.time() < t_end - 20:  # поток оборвался — дописываем остаток в следующий файл
        url = stream_url(ytdlp, vid, height)
        if url is None:
            time.sleep(30)
            continue
        out = OUT / f"{name}_{stamp}" f"{'' if part == 0 else f'_p{part}'}.mp4"
        left = int(t_end - time.time())
        log(f"{name}: пишу {left} с -> {out.name}")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", url, "-t", str(left), "-an", "-c", "copy", str(out)],
                       stdin=subprocess.DEVNULL)
        size = out.stat().st_size // (1 << 20) if out.exists() else 0
        log(f"{name}: файл {out.name}, {size} МБ")
        part += 1
    meta = OUT / f"{name}_{stamp}.json"
    meta.write_text(json.dumps({"youtube_id": vid, "name": name, "started_utc": stamp, "seconds": seconds,
                                "parts": part}, ensure_ascii=False), encoding="utf-8")
    log(f"{name}: готово")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="не ждать расписания")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--only", nargs="*", help="имена камер")
    ap.add_argument("--ytdlp", default=str(ROOT / "data" / "tools" / "ytdlp" / "Scripts" / "yt-dlp.exe"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keep_awake(True)
    now = dt.datetime.now(dt.UTC)
    threads = []
    for vid, name, tz, hhmm in CAMS:
        if a.only and name not in a.only:
            continue
        start = None if a.now else next_start_utc(tz, hhmm, now)
        th = threading.Thread(target=record, args=(a.ytdlp, vid, name, start, a.minutes * 60, a.height), daemon=True)
        th.start()
        threads.append(th)
    try:
        for th in threads:
            th.join()
    finally:
        keep_awake(False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
