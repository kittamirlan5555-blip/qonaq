"""`python -m qonaq setup` (S3): регистратор → каналы → кадры (с блюром) → конфиг объекта + .env → черновики зон.

    python -m qonaq setup                                   # спросит адрес, логин, пароль
    python -m qonaq setup --site-id cafe_arman --nvr-ip 192.168.1.64 --user admin --password '***' --vendor auto
    python -m qonaq setup --discover                        # сначала ONVIF-поиск камер в сети

Пароль пишется только в `.env` (QONAQ_NVR_PASSWORD), в YAML — ссылка ${QONAQ_NVR_PASSWORD}.
"""
from __future__ import annotations

import getpass
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from qonaq.setup.discover import (
    TEMPLATES,
    VENDOR_RU,
    ScanResult,
    build_url,
    detect_vendor,
    enc,
    onvif_discover,
    scan_channels,
)

log = logging.getLogger("qonaq.setup")


@dataclass
class FoundCamera:
    cam_id: str
    ip: str
    port: int
    vendor: str
    channel: int
    url_env: str
    snapshot: Path | None = None
    zones_path: Path | None = None
    tables: int = 0
    size: tuple[int, int] = (0, 0)
    error: str = ""
    notes: list[str] = field(default_factory=list)


def update_env(path: Path, values: dict[str, str]) -> None:
    """Обновить/добавить строки KEY=VALUE в .env, остальное не трогать."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done = set()
    for i, ln in enumerate(lines):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", ln)
        if m and m.group(1) in values:
            lines[i] = f"{m.group(1)}={values[m.group(1)]}"
            done.add(m.group(1))
    for k, v in values.items():
        if k not in done:
            lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grab_frames(url: str, n: int = 12, interval_s: float = 1.0, timeout_s: float = 10.0) -> list[np.ndarray]:
    """n кадров с шагом interval_s из потока (для черновика зон кадры должны отличаться — люди двигаются)."""
    from qonaq.sources.video import open_stream

    cap = open_stream(url, timeout_s)
    frames: list[np.ndarray] = []
    try:
        if not cap.isOpened():
            return frames
        last = 0.0
        t_end = time.time() + n * interval_s + timeout_s
        while len(frames) < n and time.time() < t_end:
            ok, f = cap.read()
            if not ok or f is None:
                time.sleep(0.05)
                continue
            now = time.time()
            if now - last >= interval_s:
                frames.append(f)
                last = now
    finally:
        cap.release()
    return frames


def process_camera(cam: FoundCamera, url: str, detector, site_id: str, out_dir: Path, zones_dir: Path,
                   n_frames: int, interval_s: float, force: bool, blur_top: float = 0.38) -> FoundCamera:
    from qonaq.detect.base import PERSON
    from qonaq.render.blur import blur_people
    from qonaq.zones.models import save_zones
    from qonaq.zones.preview import save_preview
    from qonaq.zones.suggest import suggest_from_detections

    frames = grab_frames(url, n_frames, interval_s)
    if not frames:
        cam.error = "поток открылся по RTSP, но кадры не читаются (кодек H.265? см. PILOT_RUNBOOK)"
        return cam
    h, w = frames[0].shape[:2]
    cam.size = (w, h)
    per_frame = detector.detect(frames) if detector is not None else [[] for _ in frames]
    mid = len(frames) // 2
    snap = frames[mid].copy()
    blur_people(snap, [d.xyxy for d in per_frame[mid] if d.label == PERSON], blur_top)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam.snapshot = out_dir / f"{cam.cam_id}.jpg"
    cv2.imwrite(str(cam.snapshot), snap, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if detector is None:
        return cam
    zs = suggest_from_detections(per_frame, (w, h), cam.cam_id)
    cam.tables = len(zs.zones)
    zones_dir.mkdir(parents=True, exist_ok=True)
    zp = zones_dir / f"{site_id}_{cam.cam_id}.yaml"
    if zp.exists() and not force:
        cam.notes.append(f"зоны уже есть ({zp}) — черновик рядом")
        zp = zp.with_suffix(".auto.yaml")
    save_zones(zs, zp)
    cam.zones_path = zp
    save_preview(out_dir / f"{cam.cam_id}.zones.jpg", frames[mid], zs, per_frame[mid])
    if cam.tables == 0:
        cam.notes.append("столы не найдены автоматически — разметить вручную (zones edit)")
    return cam


def _rel(p: Path) -> str:
    from qonaq.config import PROJECT_ROOT

    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def site_yaml(site_id: str, name: str, cams: list[FoundCamera], process_fps: float, model: str) -> dict:
    return {
        "site_id": site_id, "name": name, "profile": "restaurant", "process_fps": process_fps,
        "cameras": [{"id": c.cam_id, "name": f"{VENDOR_RU.get(c.vendor, c.vendor)} · канал {c.channel}",
                     "source": c.url_env,
                     **({"zones": _rel(c.zones_path)} if c.zones_path else {})} for c in cams],
        "detector": {"onnx_model": model, "onnx_tiles": 1},
        "business": {"avg_check_kzt": 15000, "open_hours": "10:00-23:00", "timezone": "Asia/Almaty"},
    }


def _ask(prompt: str, default: str = "", secret: bool = False) -> str:
    sfx = f" [{default}]" if default and not secret else ""
    v = getpass.getpass(f"{prompt}: ") if secret else input(f"{prompt}{sfx}: ")
    return v.strip() or default


def run_setup(args) -> int:
    from qonaq.config import PROJECT_ROOT, DetectorConfig

    interactive = not args.non_interactive
    site_id = args.site_id or (interactive and _ask("Код объекта (латиницей, например cafe_arman)", "site")) or "site"
    site_id = re.sub(r"[^A-Za-z0-9_]+", "_", site_id).strip("_") or "site"
    name = args.name or (interactive and _ask("Название для отчётов", site_id)) or site_id
    targets: list[str] = [args.nvr_ip] if args.nvr_ip else []
    if args.discover or (not targets and interactive):
        print("Ищу камеры и регистраторы в сети (ONVIF, 3 с)…")
        found = onvif_discover(args.discover_timeout)
        for d in found:
            print(f"  {d['ip']:15s} {d['hardware'] or ''} {d['name'] or ''}")
        if not found:
            print("  ничего не ответило (ONVIF выключен, другая подсеть или Wi-Fi изолирует клиентов)")
        if not targets:
            if interactive:
                ip = _ask("IP регистратора (или камеры)", found[0]["ip"] if found else "")
                targets = [ip] if ip else [d["ip"] for d in found]
            else:
                targets = [d["ip"] for d in found]
    if not targets:
        print("Не задан адрес: --nvr-ip 192.168.1.64 (или --discover)")
        return 2
    user = args.user or (interactive and _ask("Логин регистратора", "admin")) or "admin"
    password = args.password if args.password is not None else (
        _ask("Пароль регистратора", secret=True) if interactive else "")

    cams: list[FoundCamera] = []
    scans: list[ScanResult] = []
    for ip in targets:
        print(f"\n{ip}:{args.port} — ищу шаблон RTSP ({args.vendor})…")
        res = detect_vendor(ip, user, password, args.port, args.vendor)
        scans.append(res)
        if not res.vendor:
            print(f"  ✗ {res.error}")
            if "пароль" in res.error:
                print("  Проверьте пароль (он же — для входа в веб-интерфейс регистратора) и запустите снова.")
                return 3
            continue
        scan_channels(res, user, password, args.max_channels)
        print(f"  ✓ {VENDOR_RU[res.vendor]}: каналы {', '.join(map(str, res.channels))}")
        for ch in res.channels:
            cam_id = f"cam{ch:02d}" if len(targets) == 1 else f"cam_{ip.split('.')[-1]}_{ch:02d}"
            url_env = TEMPLATES[res.vendor].format(u="${QONAQ_NVR_USER}", p="${QONAQ_NVR_PASSWORD}", ip=ip,
                                                   port=args.port, ch=ch)
            cams.append(FoundCamera(cam_id, ip, args.port, res.vendor, ch, url_env))
    if not cams:
        print("\nКамеры не найдены. См. docs/PILOT_RUNBOOK.md, раздел «Камеры не находятся».")
        return 4

    detector = None
    if not args.no_zones:
        from qonaq.detect.base import create_detector

        detector = create_detector(DetectorConfig(onnx_model=args.model))
    runs = Path(args.out or (PROJECT_ROOT / "runs" / "setup" / site_id))
    zones_dir = PROJECT_ROOT / "configs" / "zones"
    print(f"\nСнимаю кадры ({args.frames} шт. с шагом {args.interval:g} с) и ищу столы…")
    for c in cams:
        url = build_url(c.vendor, c.ip, user, password, c.channel, c.port)
        process_camera(c, url, detector, site_id, runs, zones_dir, args.frames, args.interval, args.force)
        status = c.error or f"{c.size[0]}×{c.size[1]}, столов найдено: {c.tables}"
        print(f"  {c.cam_id} (канал {c.channel}): {status}" + "".join(f"; {n}" for n in c.notes))

    ok = [c for c in cams if not c.error]
    site_path = PROJECT_ROOT / "configs" / "sites" / f"{site_id}.yaml"
    if site_path.exists() and not args.force:
        site_path = site_path.with_suffix(".new.yaml")
    data = site_yaml(site_id, name, ok, args.fps, args.model)
    header = (f"# Создано `python -m qonaq setup` {time.strftime('%Y-%m-%d %H:%M')}. "
              "Пароль — в .env (QONAQ_NVR_PASSWORD).\n"
              f"# Дальше: поправить зоны (zones edit), сутки записи → calibrate, затем service install.\n")
    site_path.parent.mkdir(parents=True, exist_ok=True)
    site_path.write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    env_path = Path(args.env_file) if args.env_file else PROJECT_ROOT / ".env"
    update_env(env_path, {"QONAQ_NVR_USER": enc(user), "QONAQ_NVR_PASSWORD": enc(password)})
    if enc(password) != password:
        print("Внимание: в пароле есть символы @ : / ? # % — в .env он записан в URL-кодировке. Если камеры не "
              "открываются, заведите на регистраторе отдельного пользователя для Qonaq с паролем без этих символов.")
    print(f"\nКонфиг объекта: {site_path}\nЛогин и пароль: {env_path} (QONAQ_NVR_USER / QONAQ_NVR_PASSWORD)")
    print(f"Кадры и черновики зон (лица размыты): {runs}")
    print("\nДальше:")
    for c in ok:
        if c.zones_path:
            print(f"  python -m qonaq zones edit {_rel(runs / (c.cam_id + '.jpg'))} --zones {_rel(c.zones_path)}")
    print(f"  python -m qonaq live --site {_rel(site_path)}   # проверить панель")
    print(f"  python -m qonaq service install --site {_rel(site_path)}")
    return 0 if ok else 5


def add_parser(sub) -> None:
    p = sub.add_parser("setup", help="установка: найти камеры регистратора, кадры, конфиг объекта, черновики зон (S3)")
    p.add_argument("--site-id", help="код объекта латиницей (имя файла configs/sites/<код>.yaml)")
    p.add_argument("--name", help="название заведения для отчётов")
    p.add_argument("--nvr-ip", help="IP регистратора или камеры")
    p.add_argument("--port", type=int, default=554, help="порт RTSP (по умолчанию 554)")
    p.add_argument("--user", help="логин регистратора")
    p.add_argument("--password", help="пароль (лучше не указывать — спросит без отображения)")
    p.add_argument("--vendor", default="auto", choices=["auto", "hikvision", "dahua", "uniview", "xmeye"])
    p.add_argument("--max-channels", type=int, default=16)
    p.add_argument("--discover", action="store_true", help="ONVIF WS-Discovery: найти камеры в сети")
    p.add_argument("--discover-timeout", type=float, default=3.0)
    p.add_argument("--frames", type=int, default=12, help="кадров с камеры для поиска столов")
    p.add_argument("--interval", type=float, default=1.0, help="шаг между кадрами, с")
    p.add_argument("--model", default="yolox_s", help="детектор для черновика зон и для конфига")
    p.add_argument("--fps", type=float, default=2.0, help="process_fps в конфиге")
    p.add_argument("--no-zones", action="store_true", help="только найти камеры и снять кадры")
    p.add_argument("--force", action="store_true", help="перезаписать существующие конфиг и зоны")
    p.add_argument("--out", help="папка для кадров (по умолчанию runs/setup/<код>)")
    p.add_argument("--env-file", help="путь к .env (по умолчанию .env в корне проекта)")
    p.add_argument("--non-interactive", action="store_true", help="ничего не спрашивать")
