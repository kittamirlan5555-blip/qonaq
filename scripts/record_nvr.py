"""Запись каналов регистратора (NVR) по RTSP — для выезда на объект и записи 2–3 вечеров (docs/SITE_SURVEY.md §6).

    python scripts/record_nvr.py --probe                               # какие каналы есть: кадр + разрешение
    python scripts/record_nvr.py --channels 1-4 --minutes 45            # сейчас, 45 минут
    python scripts/record_nvr.py --channels 1-4 --schedule 18:00-02:00 --days 3   # каждый вечер по расписанию
    python scripts/record_nvr.py --url rtsp://... --minutes 10          # любой адрес (камера без регистратора)

Доступ — из `.env` (или переменных окружения), в командную строку пароль не пишем:
`QONAQ_NVR_IP`, `QONAQ_NVR_USER`, `QONAQ_NVR_PASSWORD`, необязательно `QONAQ_NVR_PORT` (554) и `QONAQ_NVR_VENDOR`
(hikvision | dahua | uniview | xmeye; по умолчанию — определить по каналу 1, не больше двух попыток входа, чтобы
регистратор не заблокировал пользователя). Пароль в логах и файлах скрыт.

Как пишет:
- ffmpeg без перекодирования (`-c copy`), без звука, по TCP; куски по `--segment-min` минут (15):
  `data/nvr/<ГГГГ-ММ-ДД>/chNN_<ГГГГММДД_ЧЧММСС>.mp4`. Время в имени — местное время компьютера на объекте: это
  `--video-start` для `qonaq run-file`/`billiard ledger`. Рядом — `chNN_<…>.json` с началом записи (epoch и местное)
  и списком кусков (`chNN_<…>.segments.csv` от ffmpeg: файл, начало, конец в секундах от старта).
- субпоток по умолчанию (его хватает системе и он не грузит регистратор); `--main` — основной поток (в 5–10 раз
  больше места);
- обрыв потока — переподключение через 5 с, новый кусок; меньше `--min-free-gb` (5) свободного места — стоп;
- Windows: компьютер не уходит в сон, пока идёт запись или ожидание (как у видеоплеера; настройки не меняются).

Видео — только в data/ (в .gitignore): не коммитить, не публиковать; в отчёты — кадры с блюром.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qonaq.config import load_dotenv_if_any, mask_secrets  # noqa: E402

log = logging.getLogger("record_nvr")
OUT = ROOT / "data" / "nvr"


# ------------------------------------------------------------------ чистые функции (покрыты tests/test_record_nvr.py)
def parse_channels(s: str) -> list[int]:
    """'1-4,7' -> [1, 2, 3, 4, 7]."""
    out: list[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            out += list(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main_stream(vendor: str, url: str) -> str:
    """Адрес субпотока (шаблоны qonaq/setup/discover.py) -> основной поток того же канала."""
    if vendor == "hikvision":
        return re.sub(r"(/Streaming/Channels/\d+)02\b", r"\g<1>01", url)
    if vendor == "dahua":
        return url.replace("subtype=1", "subtype=0")
    if vendor == "uniview":
        return url.replace("/s1/live", "/s0/live")
    if vendor == "xmeye":
        return url.replace("stream=1.sdp", "stream=0.sdp")
    return url


def parse_window(s: str) -> tuple[dt.time, dt.time]:
    a, b = s.split("-")
    b = b.strip().replace("24:00", "23:59")
    return dt.time.fromisoformat(a.strip().zfill(5)), dt.time.fromisoformat(b.zfill(5))


def window_at(now: dt.datetime, win: tuple[dt.time, dt.time]) -> tuple[dt.datetime, dt.datetime]:
    """Ближайшее окно записи: идущее сейчас (начало в прошлом) или следующее. Конец ≤ начала — через полночь."""
    a, b = win
    for d in (now.date() - dt.timedelta(days=1), now.date(), now.date() + dt.timedelta(days=1)):
        start = dt.datetime.combine(d, a)
        end = dt.datetime.combine(d if b > a else d + dt.timedelta(days=1), b)
        if end > now:
            return start, end
    raise AssertionError("unreachable")


def ffmpeg_cmd(exe: str, src: str, out_dir: Path, name: str, segment_s: int, duration_s: float) -> list[str]:
    cmd = [exe, "-hide_banner", "-loglevel", "error"]  # stdin открыт: «q» — дописать кусок и выйти
    if src.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp", "-timeout", "10000000"]  # 10 с без данных — обрыв
    else:  # файл (проверка без регистратора): читать с реальной скоростью, как поток
        cmd += ["-re"]
    cmd += ["-i", src, "-t", f"{max(1.0, duration_s):.0f}", "-map", "0:v:0", "-c", "copy", "-an",
            "-f", "segment", "-segment_time", str(segment_s), "-reset_timestamps", "1",
            "-segment_format", "mp4", "-segment_list", str(out_dir / f"{name}.segments.csv"),
            "-segment_list_type", "csv", "-strftime", "1", str(out_dir / f"{name.split('_')[0]}_%Y%m%d_%H%M%S.mp4")]
    return cmd


# ------------------------------------------------------------------ окружение
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("нужен ffmpeg: pip install imageio-ffmpeg или ffmpeg в PATH") from None
        return exe


def keep_awake(on: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def nvr_env() -> dict[str, str]:
    load_dotenv_if_any(Path.cwd())
    e = {k: os.environ.get(f"QONAQ_NVR_{k.upper()}", "") for k in ("ip", "user", "password", "port", "vendor")}
    e["port"] = e["port"] or "554"
    return e


def channel_urls(channels: list[int], main: bool) -> tuple[str, dict[int, str]]:
    from qonaq.setup.discover import TEMPLATES, build_url, detect_vendor

    e = nvr_env()
    if not e["ip"] or not e["user"]:
        raise SystemExit("нет QONAQ_NVR_IP / QONAQ_NVR_USER / QONAQ_NVR_PASSWORD в .env (docs/SITE_SURVEY.md §2)")
    vendor = e["vendor"]
    if vendor not in TEMPLATES:
        log.info("определяю регистратор по каналу 1 …")
        res = detect_vendor(e["ip"], e["user"], e["password"], int(e["port"]))
        if not res.vendor:
            raise SystemExit(res.error)
        vendor = res.vendor
    urls = {ch: build_url(vendor, e["ip"], e["user"], e["password"], ch, int(e["port"])) for ch in channels}
    if main:
        urls = {ch: main_stream(vendor, u) for ch, u in urls.items()}
    log.info("регистратор: %s, поток: %s", vendor, "основной" if main else "субпоток")
    return vendor, urls


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------ запись
class Channel(threading.Thread):
    """Один канал: ffmpeg до конца окна; оборвался — переподключение через retry_s, новый кусок."""

    def __init__(self, label: str, src: str, out_root: Path, end: dt.datetime, segment_s: int, stop: threading.Event,
                 retry_s: float = 5.0):
        super().__init__(name=label, daemon=True)
        self.label, self.src, self.out_root, self.end = label, src, out_root, end
        self.segment_s, self.stop, self.retry_s = segment_s, stop, retry_s
        self.starts = 0
        self.proc: subprocess.Popen | None = None

    def run(self) -> None:
        exe = ffmpeg_exe()
        while not self.stop.is_set():
            now = dt.datetime.now()
            left = (self.end - now).total_seconds()
            if left < 1:
                return
            out_dir = self.out_root / now.strftime("%Y-%m-%d")
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{self.label}_{now:%Y%m%d_%H%M%S}"
            meta = {"channel": self.label, "source": mask_secrets(self.src), "started_local": now.isoformat(
                timespec="seconds"), "started_epoch": time.time(), "segment_s": self.segment_s,
                "planned_end_local": self.end.isoformat(timespec="seconds"), "restart": self.starts}
            (out_dir / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            self.starts += 1
            log.info("%s: запись до %s → %s", self.label, self.end.strftime("%H:%M"), out_dir)
            err_path = out_dir / f"{name}.ffmpeg.log"
            with open(err_path, "w", encoding="utf-8") as err:
                self.proc = subprocess.Popen(ffmpeg_cmd(exe, self.src, out_dir, name, self.segment_s, left),
                                             stdin=subprocess.PIPE, stderr=err, text=True)
                while self.proc.poll() is None:
                    if self.stop.wait(1.0):
                        self.halt()
                        break
            lines = err_path.read_text(encoding="utf-8", errors="replace").splitlines()
            err_path.write_text(chr(10).join(mask_secrets(x) for x in lines), encoding="utf-8")  # пароль не храним
            for line in lines[-5:]:
                log.warning("%s: ffmpeg: %s", self.label, mask_secrets(line))
            if not lines:
                err_path.unlink()
            if self.stop.is_set() or (self.end - dt.datetime.now()).total_seconds() < 5:
                return
            log.warning("%s: поток оборвался (код %s) — переподключение через %.0f с", self.label,
                        self.proc.returncode, self.retry_s)
            self.stop.wait(self.retry_s)

    def halt(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            try:
                p.stdin.write("q")  # ffmpeg дописывает текущий кусок и выходит
                p.stdin.flush()
            except (OSError, ValueError):
                pass
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def record(sources: dict[str, str], end: dt.datetime, out_root: Path, segment_s: int, min_free_gb: float) -> int:
    stop = threading.Event()
    chans = [Channel(lbl, src, out_root, end, segment_s, stop) for lbl, src in sources.items()]
    for c in chans:
        c.start()
    try:
        while any(c.is_alive() for c in chans):
            if free_gb(out_root) < min_free_gb:
                log.error("свободно меньше %.0f ГБ — останавливаю запись", min_free_gb)
                stop.set()
                break
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("остановка по Ctrl+C — дописываю куски")
        stop.set()
    for c in chans:
        c.halt()
        c.join(15)
    files = sorted(out_root.rglob("*.mp4"))
    log.info("готово: %d файлов в %s", len(files), out_root)
    return 0 if files else 1


def probe(channels: list[int], out_dir: Path) -> int:
    import cv2

    from qonaq.setup.discover import ScanResult, scan_channels
    from qonaq.setup.rtsp import describe

    e = nvr_env()
    vendor, _ = channel_urls([1], False)
    res = scan_channels(ScanResult(e["ip"], int(e["port"]), vendor), e["user"], e["password"],
                        max_channels=max(channels))
    if not res.channels:
        print("каналов не найдено (проверьте пользователя «только просмотр» и RTSP в настройках регистратора)")
        return 1
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"регистратор {vendor}, каналы: {res.channels}")
    for ch in res.channels:
        from qonaq.setup.discover import build_url

        url = build_url(vendor, e["ip"], e["user"], e["password"], ch, int(e["port"]))
        r = describe(url)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok, f = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not ok:
            print(f"  канал {ch}: {r.explain()}, кадр не получен")
            continue
        p = out_dir / f"ch{ch:02d}.jpg"
        cv2.imwrite(str(p), f)
        h, w = f.shape[:2]
        warn = "  ← мало: субпоток ниже 640×360, поднять в регистраторе" if w < 640 else ""
        print(f"  канал {ch}: субпоток {w}×{h} @ {fps:.0f} к/с → {p}{warn}")
    print("Сукно на кадре: длинная сторона должна быть ≥ 150 пикселей (docs/SITE_SURVEY.md §3).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true", help="найти каналы, сохранить по кадру, показать разрешение")
    ap.add_argument("--channels", default="1-4", help="каналы регистратора: 1-4,7")
    ap.add_argument("--url", action="append", default=[], help="записать этот адрес (или файл) вместо каналов")
    ap.add_argument("--main", action="store_true", help="основной поток вместо субпотока")
    ap.add_argument("--minutes", type=float, help="записать сейчас столько минут")
    ap.add_argument("--schedule", help="окно записи ЧЧ:ММ-ЧЧ:ММ по местному времени (через полночь можно)")
    ap.add_argument("--days", type=int, default=1, help="сколько окон подряд записать (с --schedule)")
    ap.add_argument("--segment-min", type=int, default=15, help="длина куска, мин")
    ap.add_argument("--min-free-gb", type=float, default=5.0, help="меньше свободного места — стоп")
    ap.add_argument("--out", default=str(OUT), help="папка записи (по умолчанию data/nvr)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    out = Path(a.out)
    chans = parse_channels(a.channels)
    if a.probe:
        return probe(chans, out / "probe")
    if not a.minutes and not a.schedule:
        ap.error("нужно --minutes N или --schedule ЧЧ:ММ-ЧЧ:ММ")
    if a.url:
        sources = {f"url{i + 1:02d}": u for i, u in enumerate(a.url)}
    else:
        _, urls = channel_urls(chans, a.main)
        sources = {f"ch{ch:02d}": u for ch, u in urls.items()}
    log.info("источники: %s", ", ".join(f"{k}={mask_secrets(v)}" for k, v in sources.items()))
    keep_awake(True)
    try:
        if a.minutes:
            return record(sources, dt.datetime.now() + dt.timedelta(minutes=a.minutes), out, a.segment_min * 60,
                          a.min_free_gb)
        win = parse_window(a.schedule)
        rc = 0
        for i in range(a.days):
            start, end = window_at(dt.datetime.now(), win)
            wait = (start - dt.datetime.now()).total_seconds()
            if wait > 0:
                log.info("вечер %d/%d: жду начала %s (до конца — %s)", i + 1, a.days, start.strftime("%d.%m %H:%M"),
                         end.strftime("%H:%M"))
                time.sleep(wait)
            rc |= record(sources, end, out, a.segment_min * 60, a.min_free_gb)
            time.sleep(max(0.0, (end - dt.datetime.now()).total_seconds()) + 1)
        return rc
    finally:
        keep_awake(False)


if __name__ == "__main__":
    raise SystemExit(main())
