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

    p = sub.add_parser("demo", help="демо-ролик «как в рилсе»: полигоны столов + панель статусов")
    p.add_argument("video")
    p.add_argument("--out", help="папка результата (по умолчанию runs/<имя видео>/)")
    p.add_argument("--zones", help="YAML зон (по умолчанию configs/zones/<имя видео>.yaml)")
    p.add_argument("--speed", type=float, default=1.0, help="ускорение ролика (4 = в 4 раза быстрее)")
    p.add_argument("--loops", type=int, default=1, help="прогнать запись N раз подряд (для коротких записей)")
    p.add_argument("--no-blur", action="store_true", help="не размывать лица (только для отладки!)")
    p.add_argument("--snapshots", type=int, default=0)
    _add_common(p)

    p = sub.add_parser("report", help="отчёт смены из журнала событий: report.html / report.md / telegram.txt")
    p.add_argument("run_dir", help="папка прогона (runs/<имя>/)")
    p.add_argument("--site", help="YAML объекта (средний чек, пороги)")
    p.add_argument("--avg-check", type=float, help="средний чек на стол, ₸ (переопределяет конфиг)")

    p = sub.add_parser("live", help="поток (вебкамера/RTSP/файл как поток) + веб-панель + алерты")
    p.add_argument("--site", required=True, help="YAML объекта, напр. configs/sites/demo_webcam.yaml")
    p.add_argument("--source", help="переопределить источник первой камеры (webcam:1, rtsp://..., файл)")
    p.add_argument("--host", help="адрес веб-панели (по умолчанию из конфига, 127.0.0.1)")
    p.add_argument("--port", type=int, help="порт веб-панели (по умолчанию 8000)")
    p.add_argument("--no-web", action="store_true", help="без веб-панели (только события и алерты)")
    p.add_argument("--duration", type=float, help="остановиться через N секунд (для проверки)")
    p.add_argument("--fps", type=float)
    p.add_argument("--backend", choices=["auto", "ultralytics", "onnx"])
    p.add_argument("--model")
    p.add_argument("--tiles", type=int, choices=[1, 2])
    p.add_argument("--device")
    p.add_argument("--conf", type=float)

    p = sub.add_parser("eval", help="сравнить события прогона с ручной разметкой (precision/recall)")
    p.add_argument("run_dir", help="папка прогона (runs/<имя>/)")
    p.add_argument("--gt", help="YAML разметки (по умолчанию data/gt/<имя прогона>.yaml)")
    p.add_argument("--tolerance", type=float, default=10.0, help="окно совпадения по времени, с")

    rp = sub.add_parser("roles", help="роли: calibrate — калибровка формы персонала")
    r = rp.add_subparsers(dest="rcmd", required=True)
    p = r.add_parser("calibrate", help="кадры с id треков -> список id сотрудников -> профиль формы")
    p.add_argument("video")
    p.add_argument("--staff-ids", help="id треков сотрудников через запятую, напр. 3,17,21")
    p.add_argument("--auto", action="store_true", help="взять треки, побывавшие у >= 3 столов")
    p.add_argument("--out", default="configs/staff_profile.yaml", help="куда сохранить профиль формы")
    p.add_argument("--every", type=float, default=2.0, help="как часто сохранять кадр с id, с")
    p.add_argument("--rerun", action="store_true", help="заново прогнать видео, даже если tracks.json уже есть")
    _add_common(p)

    zp = sub.add_parser("zones", help="зоны: suggest | edit | preview")
    z = zp.add_subparsers(dest="zcmd", required=True)
    p = z.add_parser("suggest", help="авто-предложение столов по детекциям")
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=30, help="сколько кадров анализировать")
    p.add_argument("--out", help="куда сохранить YAML (по умолчанию configs/zones/<имя>.yaml)")
    p.add_argument("--expand", type=float, default=1.4, help="расширение bbox стола (захват стульев)")
    p.add_argument("--force", action="store_true", help="перезаписать существующий YAML")
    _add_common(p)
    p = z.add_parser("edit", help="редактор зон в окне OpenCV")
    p.add_argument("source", help="видео или картинка")
    p.add_argument("--zones", help="YAML зон (по умолчанию configs/zones/<имя>.yaml)")
    p.add_argument("--at", type=float, default=1.0, help="секунда видео для опорного кадра")
    p = z.add_parser("preview", help="PNG кадра с зонами")
    p.add_argument("source", help="видео или картинка")
    p.add_argument("--zones", help="YAML зон (по умолчанию configs/zones/<имя>.yaml)")
    p.add_argument("--out", help="PNG (по умолчанию runs/zones/<имя>.png)")
    p.add_argument("--at", type=float, default=1.0)
    p.add_argument("--no-blur", action="store_true")
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

    out = process_file(args.video, _site(args), out=args.out, mode="tracks", blur=not args.no_blur,
                       max_seconds=args.max_seconds,
                       write_video=not args.no_video, snapshots=args.snapshots)
    print(f"Результат: {out}")
    return 0


def cmd_demo(args) -> int:
    from qonaq.runner import process_file

    out = process_file(args.video, _site(args), out=args.out, mode="demo", blur=not args.no_blur,
                       speed=args.speed, max_seconds=args.max_seconds, snapshots=args.snapshots,
                       zones_path=args.zones, loops=args.loops)
    print(f"Демо-ролик: {out / 'demo.mp4'}")
    return 0


def cmd_report(args) -> int:
    from qonaq.metrics.report import generate_report

    site = _site(args)
    if args.avg_check:
        site.business.avg_check_kzt = args.avg_check
    out = generate_report(args.run_dir, site)
    print(out["telegram"].read_text(encoding="utf-8"))
    print(f"\nОтчёт: {out['html']}\n       {out['md']}\n       {out['telegram']}")
    return 0


def cmd_live(args) -> int:
    import time

    from qonaq.api.service import LiveService

    site = _site(args)
    if args.source:
        site.cameras[0].source = args.source
        site.cameras[0].realtime = True
    service = LiveService(site).start()
    host, port = args.host or site.api_host, args.port or site.api_port
    print(f"Объект: {site.name} · камер: {len(site.cameras)} · журнал: {service.run_dir / 'events.sqlite'}")
    try:
        if args.no_web:
            t0 = time.time()
            while not service.finished.is_set() and (not args.duration or time.time() - t0 < args.duration):
                time.sleep(0.2)
        else:
            import threading

            import uvicorn

            from qonaq.api.app import create_app

            print(f"Панель: http://{host}:{port}   (Ctrl+C — остановить)")
            server = uvicorn.Server(uvicorn.Config(create_app(service), host=host, port=port, log_level="warning"))
            if args.duration:
                threading.Timer(args.duration, lambda: setattr(server, "should_exit", True)).start()
            server.run()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
    if service.error:
        print(f"Ошибка пайплайна: {service.error}")
        return 1
    try:
        from qonaq.metrics.report import generate_report

        out = generate_report(service.run_dir, site)
        print(f"Отчёт: {out['html']}")
    except Exception as e:  # noqa: BLE001
        print(f"Отчёт не построен: {e}")
    return 0


def cmd_eval(args) -> int:
    from qonaq.evalgt import format_table, run_eval

    run_dir = Path(args.run_dir)
    gt = Path(args.gt) if args.gt else Path("data") / "gt" / f"{run_dir.name}.yaml"
    if not gt.exists():
        alt = Path("tests") / "fixtures" / "gt" / gt.name
        if not alt.exists():
            raise SystemExit(f"нет разметки: {gt}")
        gt = alt
    res = run_eval(run_dir, gt, args.tolerance)
    print(f"Разметка: {gt}\nОкно совпадения: ±{args.tolerance:.0f} с\n")
    print(format_table(res))
    return 0


def cmd_roles(args) -> int:
    from qonaq.roles.calibrate import build_profile, collect_tracks

    site = _site(args)
    calib = Path(site.runs_dir) / "calib" / Path(args.video).stem
    if args.rerun or not (calib / "tracks.json").exists():
        calib = collect_tracks(args.video, site, every_s=args.every, max_seconds=args.max_seconds)
        print(f"Кадры с id треков: {calib} (лица размыты). Треки: {calib / 'tracks.json'}")
    if not args.staff_ids and not args.auto:
        print("Теперь посмотрите кадры и запустите:")
        print(f"  python -m qonaq roles calibrate {args.video} --staff-ids ID1,ID2,...")
        return 0
    ids = [int(x) for x in args.staff_ids.split(",") if x.strip()] if args.staff_ids else []
    try:
        profile, used = build_profile(calib, ids, auto=args.auto, threshold=site.roles.uniform_threshold)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    out = profile.save(args.out)
    print(f"Профиль формы по трекам {used}: {out}\nПодключить: в YAML объекта  roles: {{staff_profile: {out}}}")
    return 0


def _grab_frame(source: str, at_s: float = 1.0):
    import cv2

    if Path(source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(source)
    else:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = cap.read()
        cap.release()
    if img is None:
        raise SystemExit(f"не удалось прочитать кадр из {source}")
    return img


def _zones_path(source: str, explicit: str | None) -> Path:
    return Path(explicit) if explicit else Path("configs") / "zones" / f"{Path(source).stem}.yaml"


def cmd_zones(args) -> int:
    from qonaq.zones.models import ZoneSet, load_zones, save_zones

    if args.zcmd == "suggest":
        from qonaq.detect.base import create_detector
        from qonaq.zones.preview import save_preview
        from qonaq.zones.suggest import suggest_zones

        site = _site(args)
        out = _zones_path(args.video, args.out)
        zs, frame, dets = suggest_zones(args.video, create_detector(site.detector), args.frames,
                                        expand=args.expand)
        png = Path(site.runs_dir) / "zones" / f"{Path(args.video).stem}.suggest.png"
        save_preview(png, frame, zs, dets)
        print(f"Найдено столов: {len(zs.zones)}; превью: {png}")
        if out.exists() and not args.force:
            alt = out.with_suffix(".auto.yaml")
            save_zones(zs, alt)
            print(f"{out} уже существует — черновик сохранён рядом: {alt} (--force чтобы перезаписать)")
        else:
            save_zones(zs, out)
            print(f"Зоны сохранены: {out}. Поправить: python -m qonaq zones edit {args.video}")
        return 0
    if args.zcmd == "edit":
        from qonaq.zones.editor import EditorState, run_editor

        path = _zones_path(args.source, args.zones)
        frame = _grab_frame(args.source, args.at)
        h, w = frame.shape[:2]
        zs = load_zones(path).scaled(w, h) if path.exists() else ZoneSet(camera_id=Path(args.source).stem)
        zs.frame_size = (w, h)
        print(__import__("qonaq.zones.editor", fromlist=["x"]).__doc__)
        run_editor(frame, EditorState(zs, (w, h), save_path=path))
        return 0
    if args.zcmd == "preview":
        from qonaq.detect.base import create_detector
        from qonaq.zones.preview import save_preview

        site = _site(args)
        path = _zones_path(args.source, args.zones)
        if not path.exists():
            raise SystemExit(f"нет файла зон: {path} (сначала: python -m qonaq zones suggest {args.source})")
        frame = _grab_frame(args.source, args.at)
        dets = [] if args.no_blur else create_detector(site.detector).detect([frame])[0]
        out = Path(args.out) if args.out else Path(site.runs_dir) / "zones" / f"{Path(args.source).stem}.png"
        save_preview(out, frame, load_zones(path), dets, blur=not args.no_blur)
        print(f"Превью: {out}")
        return 0
    return 1


HANDLERS = {
    "zones": cmd_zones,
    "demo": cmd_demo,
    "report": cmd_report,
    "live": cmd_live,
    "eval": cmd_eval,
    "roles": cmd_roles,
    "doctor": cmd_doctor,
    "run-file": cmd_run_file,
}

if __name__ == "__main__":
    raise SystemExit(main())
