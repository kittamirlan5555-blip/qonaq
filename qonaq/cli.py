"""Точки входа: python -m qonaq <команда>. Работает одинаково на Windows и Linux."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from qonaq.config import SiteConfig, load_dotenv_if_any, load_site


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--site", help="YAML объекта (configs/sites/*.yaml) — база для порогов и настроек")
    p.add_argument("--fps", type=float, help="сколько кадров в секунду обрабатывать (по умолчанию из конфига)")
    p.add_argument("--backend", choices=["auto", "ultralytics", "onnx"], help="детектор")
    p.add_argument("--model", help="модель: yolo11n.pt (ultralytics) или yolox_s/yolox_m (onnx)")
    p.add_argument("--tiles", type=int, choices=[1, 2], help="onnx: 2 = доп. прогон двух половин кадра")
    p.add_argument("--device", help="auto | cpu | cuda:0")
    p.add_argument("--conf", type=float, help="порог уверенности детектора")
    p.add_argument("--max-seconds", type=float, help="обработать только первые N секунд")


def _site(args) -> SiteConfig:
    site = load_site(args.site) if getattr(args, "site", None) else SiteConfig()
    if getattr(args, "fps", None):
        site.process_fps = args.fps
    d = site.detector
    if getattr(args, "backend", None):
        d.backend = args.backend
    if getattr(args, "model", None):
        if args.model.endswith(".pt"):
            d.model = args.model
        else:
            d.onnx_model = args.model
    if getattr(args, "tiles", None):
        d.onnx_tiles = args.tiles
    if getattr(args, "device", None):
        d.device = args.device
    if getattr(args, "conf", None):
        d.conf = args.conf
    return site


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qonaq", description="Qonaq Vision — видеоаналитика зала")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="проверка окружения")
    p.add_argument("--no-model", action="store_true", help="не загружать модель детекции")

    p = sub.add_parser("run-file", help="прогон видеофайла: события + аннотированное видео")
    p.add_argument("video")
    p.add_argument("--out", help="папка результата (по умолчанию runs/<имя видео>/)")
    p.add_argument("--no-blur", action="store_true", help="не размывать лица (только для отладки!)")
    p.add_argument("--no-video", action="store_true", help="не писать видео, только события")
    p.add_argument("--snapshots", type=int, default=0, help="сохранить N кадров-примеров (с блюром)")
    _add_common(p)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for stream in (sys.stdout, sys.stderr):  # кириллица в консоли Windows
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    load_dotenv_if_any(Path.cwd())
    handler = HANDLERS[args.cmd]
    return int(handler(args) or 0)


def cmd_doctor(args) -> int:
    from qonaq import doctor

    return doctor.run(check_model=not args.no_model)


def cmd_run_file(args) -> int:
    from qonaq.runner import process_file

    out = process_file(args.video, _site(args) if (args.site or args.fps or args.model or args.backend
                                                    or args.tiles or args.device or args.conf) else None,
                       out=args.out, mode="tracks", blur=not args.no_blur, max_seconds=args.max_seconds,
                       write_video=not args.no_video, snapshots=args.snapshots)
    print(f"Результат: {out}")
    return 0


HANDLERS = {
    "doctor": cmd_doctor,
    "run-file": cmd_run_file,
}

if __name__ == "__main__":
    raise SystemExit(main())
