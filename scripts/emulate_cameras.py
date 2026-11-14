"""Эмуляция регистратора: ролики из data/videos отдаются по RTSP зацикленно, с логином/паролем (SPRINT2 S1).

Нужно, чтобы проверить `live` на «камерах» без камер: многокамерный режим, долгий прогон, обрыв и
переподключение. Работает на Windows, Linux и macOS — только Python, MediaMTX и ffmpeg.

    python scripts/emulate_cameras.py                      # 4 камеры, rtsp://admin:qonaq12345@127.0.0.1:8554/...
    python scripts/emulate_cameras.py --outage-at 1200 --outage-s 120   # через 20 мин «выдернуть» сервер на 2 мин
    python -m qonaq live --site configs/sites/rtsp_emulated.yaml --no-web --duration 3900

Что делает:
1. MediaMTX (MIT, github.com/bluenviron/mediamtx) — качает релиз под вашу ОС в data/tools/mediamtx/, сверяет
   SHA-256 с checksums.sha256 того же релиза. Или укажите свой бинарник: --mediamtx путь.
2. Готовит «субпоток» каждой камеры один раз: H.264, ширина --width, --fps кадр/с (как 102-й канал Hikvision),
   кэш в data/emulator/. Дальше ffmpeg публикует его без перекодирования (почти не ест CPU) по кругу.
3. Пути как у настоящих регистраторов: Hikvision `/Streaming/Channels/<N>02`, Dahua
   `/cam/realmonitor?channel=1&subtype=1` (MediaMTX не учитывает query: путь — `cam/realmonitor`, параметры
   принимаются и игнорируются, поэтому Dahua-канал в эмуляции один).
4. Следит за публикаторами: если сервер упал/перезапущен — поднимает их заново.

Ничего из data/ не коммитится (.gitignore).
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIAMTX_VERSION = "v1.21.2"
RELEASE = "https://github.com/bluenviron/mediamtx/releases/download/{v}/{name}"

DEFAULT_CAMS = [
    # (видео, путь на «регистраторе», вид)
    ("cctv_restaurant_corner.mp4", "Streaming/Channels/102", "hikvision"),
    ("cctv_terrace_cafe.mov", "Streaming/Channels/202", "hikvision"),
    ("topdown_cafe_people.mp4", "Streaming/Channels/302", "hikvision"),
    ("foodcourt_preannotated.mp4", "cam/realmonitor", "dahua"),
]


@dataclass
class Cam:
    video: Path
    path: str
    vendor: str
    prepared: Path | None = None
    proc: subprocess.Popen | None = None

    def url(self, host: str, port: int, user: str, password: str) -> str:
        q = "?channel=1&subtype=1" if self.vendor == "dahua" else ""
        return f"rtsp://{user}:{password}@{host}:{port}/{self.path}{q}"


def asset_name(version: str) -> str:
    sysname = platform.system().lower()
    mach = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(mach, mach)
    if sysname == "windows":
        return f"mediamtx_{version}_windows_amd64.zip"
    if sysname == "darwin":
        return f"mediamtx_{version}_darwin_{arch}.tar.gz"
    if arch == "arm64":
        return f"mediamtx_{version}_linux_arm64.tar.gz"
    return f"mediamtx_{version}_linux_{arch}.tar.gz"


def ensure_mediamtx(version: str, dest: Path) -> Path:
    exe = dest / ("mediamtx.exe" if platform.system() == "Windows" else "mediamtx")
    if exe.exists():
        return exe
    dest.mkdir(parents=True, exist_ok=True)
    name = asset_name(version)
    arc = dest / name
    print(f"качаю MediaMTX {version}: {name}")
    urllib.request.urlretrieve(RELEASE.format(v=version, name=name), arc)  # noqa: S310 — фиксированный https
    sums = dest / "checksums.sha256"
    urllib.request.urlretrieve(RELEASE.format(v=version, name="checksums.sha256"), sums)  # noqa: S310
    want = next((ln.split()[0] for ln in sums.read_text().splitlines() if ln.strip().endswith(name)), None)
    got = hashlib.sha256(arc.read_bytes()).hexdigest()
    if want is None or want.lower() != got:
        arc.unlink(missing_ok=True)
        raise SystemExit(f"контрольная сумма {name} не совпала (ожидалась {want}, получена {got})")
    if name.endswith(".zip"):
        with zipfile.ZipFile(arc) as z:
            z.extractall(dest)
    else:
        with tarfile.open(arc) as t:
            t.extractall(dest, filter="data")
    exe.chmod(0o755)
    return exe


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("нет ffmpeg: pip install imageio-ffmpeg") from None
        return exe


def prepare(cam: Cam, out_dir: Path, width: int, fps: float, ff: str) -> Path:
    """Один раз перекодировать ролик в «субпоток» (H.264, GOP = 2 с) — дальше публикуем без перекодирования."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cam.video.stem}_{width}w_{fps:g}fps.mp4"
    if out.exists() and out.stat().st_size > 10_000:
        return out
    print(f"готовлю субпоток {out.name} ...")
    gop = max(1, int(round(fps * 2)))
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(cam.video), "-an",
           "-vf", f"scale={width}:-2,fps={fps:g}", "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "main",
           "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
           "-b:v", "1200k", str(out)]
    subprocess.run(cmd, check=True)
    return out


def write_config(path: Path, port: int, user: str, password: str) -> None:
    """Минимальный mediamtx.yml: только RTSP (TCP+UDP), логин/пароль на чтение, публикация — с localhost."""
    path.write_text(f"""# сгенерировано scripts/emulate_cameras.py
logLevel: warn
rtsp: true
rtspAddress: :{port}
rtspTransports: [tcp, udp]
rtspEncryption: "no"
rtmp: false
hls: false
webrtc: false
srt: false
api: false
metrics: false
pprof: false
playback: false
authMethod: internal
authInternalUsers:
  - user: {user}
    pass: {password}
    ips: []
    permissions:
      - action: read
      - action: publish
  - user: any
    pass:
    ips: ["127.0.0.1", "::1"]
    permissions:
      - action: publish
paths:
  all_others:
""", encoding="utf-8")


def start_publisher(cam: Cam, ff: str, port: int, user: str, password: str) -> subprocess.Popen:
    url = f"rtsp://{user}:{password}@127.0.0.1:{port}/{cam.path}"
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(cam.prepared),
           "-c", "copy", "-an", "-f", "rtsp", "-rtsp_transport", "tcp", url]
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)


def start_server(exe: Path, cfg: Path) -> subprocess.Popen:
    return subprocess.Popen([str(exe), str(cfg)], cwd=str(cfg.parent), stdin=subprocess.DEVNULL)


def stop(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(5)
    except subprocess.TimeoutExpired:
        p.kill()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", nargs="*", help="ролики (по умолчанию 4 из data/videos)")
    ap.add_argument("--host", default="127.0.0.1", help="адрес в печатаемых URL")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="qonaq12345")
    ap.add_argument("--width", type=int, default=960, help="ширина субпотока (по умолчанию 960)")
    ap.add_argument("--fps", type=float, default=12.0, help="кадр/с субпотока (по умолчанию 12)")
    ap.add_argument("--mediamtx", help="путь к своему бинарнику MediaMTX")
    ap.add_argument("--version", default=MEDIAMTX_VERSION)
    ap.add_argument("--duration", type=float, default=0, help="остановиться через N секунд (0 — до Ctrl+C)")
    ap.add_argument("--outage-at", type=float, default=0, help="через N секунд остановить сервер (обрыв)")
    ap.add_argument("--outage-s", type=float, default=120, help="длительность обрыва, с")
    args = ap.parse_args(argv)

    vdir = ROOT / "data" / "videos"
    if args.videos:
        cams = [Cam(Path(v), f"Streaming/Channels/{i + 1}02", "hikvision") for i, v in enumerate(args.videos)]
    else:
        cams = [Cam(vdir / v, p, k) for v, p, k in DEFAULT_CAMS]
    missing = [str(c.video) for c in cams if not c.video.exists()]
    if missing:
        raise SystemExit("нет роликов: " + ", ".join(missing) + " — запустите python scripts/fetch_videos.py")
    ff = ffmpeg_exe()
    work = ROOT / "data" / "emulator"
    for c in cams:
        c.prepared = prepare(c, work, args.width, args.fps, ff)
    exe = Path(args.mediamtx) if args.mediamtx else ensure_mediamtx(args.version, ROOT / "data" / "tools" / "mediamtx")
    cfg = work / "mediamtx.yml"
    write_config(cfg, args.port, args.user, args.password)

    server = start_server(exe, cfg)
    time.sleep(1.0)
    print("Камеры:")
    for i, c in enumerate(cams, 1):
        print(f"  cam{i} [{c.vendor}] {c.video.name}: {c.url(args.host, args.port, args.user, args.password)}")
    sys.stdout.flush()

    stopping = {"v": False}

    def _sig(*_):
        stopping["v"] = True

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)
    t0 = time.time()
    outage_done = args.outage_at <= 0
    down_until = 0.0
    try:
        while not stopping["v"]:
            now = time.time() - t0
            if args.duration and now >= args.duration:
                break
            if not outage_done and now >= args.outage_at and server is not None:
                print(f"[{time.strftime('%H:%M:%S')}] ОБРЫВ: сервер остановлен на {args.outage_s:.0f} с", flush=True)
                for c in cams:
                    stop(c.proc)
                    c.proc = None
                stop(server)
                server = None
                down_until = now + args.outage_s
                outage_done = True
            if server is None and now >= down_until:
                print(f"[{time.strftime('%H:%M:%S')}] сервер снова запущен", flush=True)
                server = start_server(exe, cfg)
                time.sleep(1.0)
            if server is not None:
                if server.poll() is not None:  # сервер умер сам — перезапуск
                    print("сервер завершился, перезапускаю", flush=True)
                    server = start_server(exe, cfg)
                    time.sleep(1.0)
                for c in cams:
                    if c.proc is None or c.proc.poll() is not None:
                        c.proc = start_publisher(c, ff, args.port, args.user, args.password)
            time.sleep(1.0)
    finally:
        for c in cams:
            stop(c.proc)
        stop(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
