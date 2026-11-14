"""Точки входа: python -m qonaq <команда>. Работает одинаково на Windows и Linux."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from qonaq.config import SiteConfig, load_dotenv_if_any, load_site


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--site", help="YAML объекта (configs/sites/*.yaml) — база для порогов и настроек")
    p.add_argument("--fps", type=float, help="сколько кадров в секунду обрабатывать (по умолчанию из конфига)")
    p.add_argument("--backend", choices=["auto", "ultralytics", "onnx"], help="детектор")
    p.add_argument("--model", help="модель: yolo11n.pt (ultralytics) или yolox_s/yolox_m (onnx)")
    p.add_argument("--tiles", type=int, choices=[1, 2, 3, 4], help="onnx: 2 = две половины кадра; 3–4 — сетка")
    p.add_argument("--device", help="auto | cpu | cuda:0")
    p.add_argument("--conf", type=float, help="порог уверенности детектора")
    p.add_argument("--max-seconds", type=float, help="обработать только первые N секунд")
    p.add_argument("--det-cache", help="папка кэша детекций (повторные прогоны того же видео — без детектора)")
    p.add_argument("--roles-head", help="голова «персонал/гость» (models/roles_head.npz); 'none' — выключить")


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
    if getattr(args, "det_cache", None):
        d.cache_dir = args.det_cache
    rh = getattr(args, "roles_head", None)
    if rh:
        site.roles.classifier = None if rh.lower() == "none" else rh
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
    p.add_argument("run_dir", help="папка прогона (runs/<имя>/) или слово pilot — отчёт пилота «как есть / стало»")
    p.add_argument("--site", help="YAML объекта (средний чек, пороги)")
    p.add_argument("--avg-check", type=float, help="средний чек на стол, ₸ (переопределяет конфиг)")
    p.add_argument("--baseline", nargs="+", help="pilot: журналы «как есть» (папки прогонов / events.sqlite)")
    p.add_argument("--baseline-from", help="pilot: начало периода «как есть», ГГГГ-ММ-ДД[ ЧЧ:ММ]")
    p.add_argument("--baseline-to", help="pilot: конец периода «как есть» (дата включительно)")
    p.add_argument("--after", nargs="+", help="pilot: журналы «стало»")
    p.add_argument("--after-from", help="pilot: начало периода «стало»")
    p.add_argument("--after-to", help="pilot: конец периода «стало»")
    p.add_argument("--out", help="pilot: куда сохранить HTML (по умолчанию runs/pilot_report.html)")

    p = sub.add_parser("demo-reel", help="ролик для продаж 30–45 с: синтетика полного цикла + реальный CCTV (S6)")
    p.add_argument("--scenario", default="configs/synth/demo_reel.yaml")
    p.add_argument("--real", default="data/videos/cctv_terrace_cafe.mov", help="реальная запись для вставки")
    p.add_argument("--real-seconds", type=float, default=8.0)
    p.add_argument("--real-from", type=float, default=36.0,
                   help="с какой секунды реальной записи брать кусок (раньше статусы ещё не установились)")
    p.add_argument("--out", default="runs/demo_reel.mp4")
    p.add_argument("--rerun", action="store_true", help="пересчитать синтетику и реальный кусок заново")
    p.add_argument("--billiard", action="store_true",
                   help="ролик бильярдной (B6): процедурный зал → профиль billiard → строка сверки с учётом")
    p.add_argument("--schematic", action="store_true",
                   help="ресторан без чужих кадров (B6): схема зала сверху, фигуры-схемы; можно показывать")
    _add_common(p)

    p = sub.add_parser("synth-week", help="синтетическая неделя зала (S5): параметры YAML -> журнал событий")
    p.add_argument("params", nargs="+", help="configs/synth_week/*.yaml")
    p.add_argument("--out", default="runs/pilot_synth", help="папка; журнал каждой недели — <out>/<name>/")
    p.add_argument("--site", help="YAML объекта (пороги)")

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
    p.add_argument("--tiles", type=int, choices=[1, 2, 3, 4])
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

    p = sub.add_parser("synth", help="синтетика полного цикла (W4): сценарий YAML -> ролик + разметка -> прогон + eval")
    p.add_argument("scenarios", nargs="*", help="configs/synth/*.yaml (по умолчанию все, кроме _*)")
    p.add_argument("--render-only", action="store_true", help="только ролик и разметка, без прогона")
    p.add_argument("--no-render", action="store_true", help="ролик уже есть — только прогон и eval")
    p.add_argument("--demo", action="store_true", help="писать демо-ролик (медленнее)")
    _add_common(p)

    p = sub.add_parser("record", help="запись вебкамеры/потока в файл для постановочного теста (W5)")
    p.add_argument("--source", default="webcam:0")
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--out", default="data/videos/staged.mp4")
    p.add_argument("--rec-fps", type=float, default=15.0, help="частота записи")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--preview", action="store_true", help="окно с картинкой (q — стоп)")

    p = sub.add_parser("staged-check", help="постановочный тест одной строкой: зоны, пайплайн, eval, демо (W5)")
    p.add_argument("video")
    p.add_argument("--gt", help="разметка (по умолчанию data/gt/<имя>.yaml)")
    _add_common(p)

    cp = sub.add_parser("calibrate", help="калибровка «персонал/гость» под зал: collect | review | train | promote | "
                                          "rollback | status | curve (W6)")
    c = cp.add_subparsers(dest="cal_cmd", required=True)
    p = c.add_parser("collect", help="прогон записи зала -> кропы (с блюром) + признаки + авторазметка")
    p.add_argument("video")
    p.add_argument("--site-id", required=True)
    p.add_argument("--guest-min-s", type=float, default=300, help="сидел в зоне стола не меньше — уверенный гость")
    p.add_argument("--staff-min-tables", type=int, default=3, help="стоя у стольких столов — уверенный персонал")
    p.add_argument("--every-s", type=float, default=2.0, help="как часто брать кроп трека")
    _add_common(p)
    p = c.add_parser("review", help="локальная страница проверки неуверенных кропов")
    p.add_argument("--site-id", required=True)
    p.add_argument("--port", type=int, default=8010)
    p = c.add_parser("train", help="новая голова зала: общие датасеты + кропы зала; «было → стало»")
    p.add_argument("--site-id", required=True)
    p.add_argument("--with-general", action="store_true", help="подмешать общие датасеты (по умолчанию нет, D-022)")
    p = c.add_parser("promote", help="включить версию головы (по умолчанию последнюю)")
    p.add_argument("--site-id", required=True)
    p.add_argument("--version", type=int)
    p = c.add_parser("rollback", help="вернуть прошлую включённую версию")
    p.add_argument("--site-id", required=True)
    p = c.add_parser("status", help="версии и какая включена")
    p.add_argument("--site-id", required=True)
    p = c.add_parser("curve", help="кривая «n размеченных кропов -> точность» на отложенных залах W2")
    p.add_argument("--ns", default="0,25,50,100,300")

    p = sub.add_parser("health", help="здоровье прогона/живой системы: проблемы, сводка (W7)")
    p.add_argument("run_dir")
    p.add_argument("--daily", action="store_true", help="собрать ежедневную сводку (и отправить владельцу, если "
                                                        "есть TELEGRAM_OWNER_CHAT_ID)")
    p.add_argument("--site")

    dp = sub.add_parser("data", help="обучающие датасеты: index — единая схема и разбиения (W1)")
    d = dp.add_subparsers(dest="data_cmd", required=True)
    p = d.add_parser("index", help="data/datasets -> data/index/{roles,tables}.csv + summary.json")
    p.add_argument("--root", default="data/datasets")
    p.add_argument("--out", default="data/index")
    p.add_argument("--no-mine", action="store_true", help="не добирать гостей зала C детектором")

    tp = sub.add_parser("train", help="обучение/оценка лёгких голов (нужен scikit-learn)")
    t = tp.add_subparsers(dest="train_cmd", required=True)
    p = t.add_parser("roles", help="персонал/гость: leave-one-venue-out + итоговая голова models/roles_head.npz")
    p.add_argument("--index", default="data/index/roles.csv")
    p.add_argument("--encoders", default="mnv2,reid")
    p.add_argument("--final", help="вариант признаков для итоговой головы, напр. mnv2+hsv")
    p.add_argument("--out", default="runs/train")
    p.add_argument("--models", default="models")
    p = t.add_parser("tables", help="стол clean/dirty/occupied: по камерам + внешний тест, models/tables_head.npz")
    p.add_argument("--index", default="data/index/tables.csv")
    p.add_argument("--out", default="runs/train")
    p.add_argument("--models", default="models")

    from qonaq.alerts.cli_telegram import add_parser as add_telegram
    from qonaq.billiard.cli import add_parser as add_billiard
    from qonaq.setup.service import add_parser as add_service
    from qonaq.setup.wizard import add_parser as add_setup

    add_telegram(sub)
    add_setup(sub)
    add_service(sub)
    add_billiard(sub)
    from qonaq.reconcile import add_parser as add_reconcile

    add_reconcile(sub)
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
    logging.getLogger("httpx").setLevel(logging.WARNING)  # не печатать каждый запрос к Telegram
    log_file = os.environ.get("QONAQ_LOG_FILE")
    if log_file:  # служба (S3): лог в файл с ротацией 10 МБ × 5
        from logging.handlers import RotatingFileHandler

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=10 * 2**20, backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(fh)
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
    if args.run_dir == "pilot":
        from qonaq.metrics.pilot import generate_pilot_report, summary_lines

        if not args.baseline:
            raise SystemExit("нужно --baseline <журналы> (и обычно --after <журналы>)")
        out, a, b = generate_pilot_report(site, args.baseline, args.after, args.out or "runs/pilot_report.html",
                                          args.baseline_from, args.baseline_to, args.after_from, args.after_to)
        print("\n".join(summary_lines(a, b, site)))
        print(f"\nОтчёт пилота: {out}")
        return 0
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
            import signal

            stop = {"v": False}

            def _term(*_):  # служба/kill: штатная остановка с записью health.json, run.json и отчёта
                stop["v"] = True

            for sig in ("SIGTERM", "SIGINT", "SIGBREAK"):
                if hasattr(signal, sig):
                    try:
                        signal.signal(getattr(signal, sig), _term)
                    except (ValueError, OSError):
                        pass
            t0 = time.time()
            while not stop["v"] and not service.finished.is_set() and (
                    not args.duration or time.time() - t0 < args.duration):
                time.sleep(0.2)
        else:
            import threading

            import uvicorn

            from qonaq.api.app import create_app, graceful_server

            print(f"Панель: http://{host}:{port}   (Ctrl+C — остановить)")
            server = graceful_server(uvicorn.Config(create_app(service), host=host, port=port, log_level="warning"))
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


def cmd_synth(args) -> int:
    from qonaq.config import PROJECT_ROOT
    from qonaq.evalgt import format_table, run_eval
    from qonaq.runner import process_file
    from qonaq.synth.render import build
    from qonaq.synth.scenario import Scenario

    paths = [Path(p) for p in args.scenarios] or sorted(
        p for p in (PROJECT_ROOT / "configs" / "synth").glob("*.yaml") if not p.name.startswith("_"))
    rows = []
    for sp in paths:
        sc = Scenario.load(sp)
        name = f"synth_{sc.name}"
        video = PROJECT_ROOT / "data" / "videos" / f"{name}.mp4"
        if not args.no_render or not video.exists():
            build(sp, PROJECT_ROOT)
        if args.render_only:
            continue
        site = _site(args)
        if not args.fps:
            site.process_fps = sc.fps
        if not args.site:
            # ролик начинается с пустого зала: правило «сидели до старта» (D-010) не нужно
            site.thresholds.preexisting_grace_s = 0
        run_dir = Path(site.runs_dir) / name
        process_file(str(video), site, out=str(run_dir), mode="demo" if args.demo else "tracks", blur=True,
                     max_seconds=args.max_seconds, write_video=args.demo, snapshots=0)
        res = run_eval(run_dir, PROJECT_ROOT / "data" / "gt" / f"{name}.yaml")
        print(f"\n## {name}: {sc.description}\n")
        print((run_dir / "eval.md").read_text(encoding="utf-8"))
        tot = res["total"]
        rows.append((name, tot, res.get("alerts")))
    if rows:
        lines = ["| Сценарий | событий в разметке | у системы | совпало | precision | recall | алерты |",
                 "|---|---|---|---|---|---|---|"]
        for name, t, al in rows:
            a = "—" if al is None else ("OK" if al["ok"] else "НЕ OK") + " (" + (
                ", ".join(f"{x['kind']}@{x['t']:.0f}" for x in al["all"]) or "нет") + ")"
            lines.append(f"| {name} | {t['gt']} | {t['system']} | {t['tp']} | {t['precision']} | {t['recall']} | {a} |")
        out = Path(_site(args).runs_dir) / "synth_summary.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(lines))
        _ = format_table
    return 0


def cmd_record(args) -> int:
    from qonaq.sources.record import record

    res = record(args.source, args.seconds, Path(args.out), fps=args.rec_fps, width=args.width,
                 height=args.height, preview=args.preview)
    print(f"Записано: {res['out']} — {res['seconds']} с, {res['size'][0]}x{res['size'][1]}")
    return 0


def cmd_staged(args) -> int:
    from qonaq.staged import staged_check

    def overrides(site):
        base = _site(args)  # флаги --model/--fps/... поверх configs/sites/staged.yaml
        if args.fps:
            site.process_fps = base.process_fps
        for k in ("backend", "onnx_model", "onnx_tiles", "model", "device", "conf", "cache_dir"):
            if getattr(args, {"onnx_model": "model", "onnx_tiles": "tiles", "cache_dir": "det_cache"}.get(k, k), None):
                setattr(site.detector, k, getattr(base.detector, k))
        if args.roles_head:
            site.roles.classifier = base.roles.classifier

    return staged_check(args.video, args.gt, site_path=args.site, overrides=overrides)


def cmd_calibrate(args) -> int:
    import json

    from qonaq.calib import store

    sid = getattr(args, "site_id", None)
    if args.cal_cmd == "collect":
        from qonaq.calib.collect import collect

        res = collect(args.video, _site(args), sid, every_s=args.every_s, guest_min_s=args.guest_min_s,
                      staff_min_tables=args.staff_min_tables, max_seconds=args.max_seconds)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"Дальше: python -m qonaq calibrate review --site-id {sid}")
        return 0
    if args.cal_cmd == "review":
        from qonaq.calib.review import serve

        serve(sid, port=args.port)
        return 0
    if args.cal_cmd == "train":
        from qonaq.calib.train import format_report, train_site

        print(format_report(train_site(sid, use_general_data=args.with_general)))
        return 0
    if args.cal_cmd == "promote":
        v = store.promote(sid, args.version)
        print(f"Включена v{v}: {store.active_path(sid)}. В конфиге объекта: roles.classifier: "
              f"models/site/{sid}/active.npz")
        return 0
    if args.cal_cmd == "rollback":
        v = store.rollback(sid)
        print(f"Включена v{v}" if v else "Голова зала выключена (active.npz удалён) — роли по поведению/общей голове")
        return 0
    if args.cal_cmd == "status":
        print(json.dumps({"versions": store.versions(sid), "active": store.active_version(sid),
                          "history": store.history(sid)}, ensure_ascii=False, indent=1))
        return 0
    if args.cal_cmd == "curve":
        from qonaq.calib.train import calibration_curve, curve_markdown
        from qonaq.train.featcache import extract

        f = {"mnv2": extract(Path("data/index/roles.csv"), "mnv2", Path("data/features"))}
        c = calibration_curve(f, ns=tuple(int(x) for x in args.ns.split(",")))
        out = Path("runs/train")
        out.mkdir(parents=True, exist_ok=True)
        (out / "calib_curve.json").write_text(json.dumps(c, indent=1, default=float), encoding="utf-8")
        (out / "calib_curve.md").write_text(curve_markdown(c), encoding="utf-8")
        print(curve_markdown(c))
        return 0
    return 1


def cmd_health(args) -> int:
    import json

    from qonaq.events.store import EventStore
    from qonaq.health.monitor import check_events

    run_dir = Path(args.run_dir)
    saved = json.loads((run_dir / "health.json").read_text(encoding="utf-8")) if (run_dir / "health.json").exists() \
        else {"status": "ok", "issues": [], "cameras": {}}
    store = EventStore(run_dir / "events.sqlite")
    try:
        events = store.events()
        fb = store.feedback()
    finally:
        store.close()
    seen = {(i["kind"], i["camera_id"], i["zone_id"]) for i in saved["issues"]}
    for i in check_events(events):  # проверки по журналу (на случай, если health.json старый)
        if (i.kind, i.camera_id, i.zone_id) not in seen:
            saved["issues"].append(i.to_dict())
    if saved["issues"] and saved["status"] == "ok":
        saved["status"] = "error" if any(i["severity"] == "error" for i in saved["issues"]) else "warn"
    print(f"Статус: {saved['status']}")
    for i in saved["issues"]:
        print(f"  [{i['severity']}] {i['camera_id']} {i['zone_id']}: {i['message']}")
    for cid, c in saved.get("cameras", {}).items():
        print(f"  {cid}: посадок {c.get('seated')}, визитов персонала {c.get('visits')}, "
              f"unknown {100 * c.get('unknown_share', 0):.0f}%")
    if fb:
        print(f"Отметок «неверно»: {len(fb)}")
    from qonaq.health.downtime import downtime, downtime_lines

    tz = None
    if args.site:
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(load_site(args.site).business.timezone)
        except Exception:  # noqa: BLE001
            tz = None
    gaps = downtime(events)  # по журналу — переживает перезапуски (health.json — только последний процесс)
    if gaps:
        print("Простои по журналу (heartbeat):")
        for ln in downtime_lines(gaps, tz, limit=20):
            print(f"  {ln}")
    if args.daily:
        from qonaq.alerts.telegram import sender_from_env
        from qonaq.health.daily import build_daily, send_daily

        site = load_site(args.site) if args.site else SiteConfig()
        frames = []
        for f in sorted((run_dir / "frames").glob("*.jpg"))[: site.health.daily_frames]:
            frames.append((f.read_bytes(), f.stem))  # снимки прогона уже с блюром
        daily = build_daily(run_dir, site.name, saved, frames)
        print("\n" + daily["text"])
        owner = sender_from_env(site.health.owner_chat_env)
        if owner is not None:
            send_daily(owner, daily)
            print("Отправлено владельцу системы.")
        else:
            print(f"(не отправлено: нет TELEGRAM_BOT_TOKEN / {site.health.owner_chat_env}); файлы: {daily['dir']}")
    return 0 if saved["status"] != "error" else 2


def cmd_data(args) -> int:
    import json

    from qonaq.config import PROJECT_ROOT
    from qonaq.datasets.index import build

    if args.data_cmd == "index":
        summary = build(Path(args.root), PROJECT_ROOT, Path(args.out), mine=not args.no_mine)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    return 1


def cmd_train(args) -> int:
    import json

    if args.train_cmd == "roles":
        from qonaq.train.roles_eval import run

        res = run(Path(args.index), [e for e in args.encoders.split(",") if e], Path(args.out), Path(args.models),
                  final=args.final)
        print((Path(args.out) / "roles_eval.md").read_text(encoding="utf-8"))
        print(json.dumps({"head": res["head"], "final_cfg": res["final_cfg"]}, ensure_ascii=False))
        return 0
    if args.train_cmd == "tables":
        from qonaq.train.tables_eval import run as run_tables

        res = run_tables(Path(args.index), Path(args.out), Path(args.models))
        print((Path(args.out) / "tables_eval.md").read_text(encoding="utf-8"))
        print(json.dumps({"head": res["head"], "best": res["best"]}, ensure_ascii=False))
        return 0
    return 1


def cmd_telegram(args) -> int:
    from qonaq.alerts.cli_telegram import run

    return run(args)


def cmd_setup(args) -> int:
    from qonaq.setup.wizard import run_setup

    return run_setup(args)


def cmd_service(args) -> int:
    from qonaq.setup.service import run

    return run(args)


def _billiard_reel(args) -> int:
    from datetime import datetime

    import yaml

    from qonaq.billiard.cli import run_video
    from qonaq.billiard.synth import build
    from qonaq.config import PROJECT_ROOT
    from qonaq.events.store import EventStore
    from qonaq.reconcile import Settings, local_tz, read_paid_csv, reconcile, telegram_text, video_sessions
    from qonaq.render.reel import billiard_keys, build_billiard_reel

    scen = PROJECT_ROOT / "configs" / "synth_billiard" / "demo_reel.yaml"
    spec = yaml.safe_load(scen.read_text(encoding="utf-8"))
    name = f"synth_{spec['name']}"
    vdir = PROJECT_ROOT / "data" / "videos"
    video, zones = vdir / f"{name}.mp4", vdir / f"{name}.yaml"
    if args.rerun or not video.exists():
        build(scen, vdir)
    run = Path("runs") / name
    if args.rerun or not (run / "demo.mp4").exists():
        run_video(video, zones, run, demo=True)
    tz = local_tz()
    start = datetime(2026, 11, 14, 20, 0, tzinfo=tz).timestamp()
    st = EventStore(run / "events.sqlite")
    paid, _ = read_paid_csv(PROJECT_ROOT / "configs" / "synth_billiard" / "demo_reel_paid.csv",
                            datetime(2026, 11, 14).date(), tz)
    res = reconcile(video_sessions(st, "billiard", start), paid, Settings(profile="billiard"),
                    [(start, start + spec["duration_s"])])
    lean = [(ev["table"], float(ev["lean"][0]) + 3) for ev in spec["timeline"] if "lean" in ev]
    keys = billiard_keys(st, lean, paid={"B1"}, names={"B1": "Стол 1", "B2": "Стол 2"})
    st.close()
    lines = [x.strip() for x in telegram_text(res, "", tz).splitlines()[1:]][:4]
    out = Path(args.out if args.out != "runs/demo_reel.mp4" else "runs/demo_reel_billiard.mp4")
    build_billiard_reel(run, out, keys, lines, scene_fps=float(spec["fps"]))
    print(f"Демо-ролик бильярда: {out}")
    return 0


def cmd_demo_reel(args) -> int:
    if getattr(args, "billiard", False):
        return _billiard_reel(args)
    if getattr(args, "schematic", False):
        from qonaq.render.schematic import cmd_schematic_reel

        return cmd_schematic_reel(args)
    from qonaq.config import PROJECT_ROOT
    from qonaq.render.reel import build_reel
    from qonaq.runner import process_file
    from qonaq.synth.render import build
    from qonaq.synth.scenario import Scenario

    sp = Path(args.scenario)
    sc = Scenario.load(sp)
    name = f"synth_{sc.name}"
    video = PROJECT_ROOT / "data" / "videos" / f"{name}.mp4"
    if args.rerun or not video.exists():
        if not (PROJECT_ROOT / "data" / "index" / "roles.csv").exists():  # синтетике нужны фигуры из датасетов
            print("Для demo-reel нужны датасеты и индекс (~1 ГБ, один раз):\n"
                  "  python scripts/fetch_datasets.py\n  python -m qonaq data index")
            return 2
        build(sp, PROJECT_ROOT)
    site = _site(args)
    site.process_fps = sc.fps
    site.thresholds.preexisting_grace_s = 0
    synth_run = Path(site.runs_dir) / name
    if args.rerun or not (synth_run / "demo.mp4").exists():
        process_file(str(video), site, out=str(synth_run), mode="demo", blur=True, write_video=True)
    real_run = None
    real = Path(args.real)
    if real.exists():
        real_run = Path(site.runs_dir) / f"reel_{real.stem}"
        if args.rerun or not (real_run / "demo.mp4").exists():
            rs = _site(args)
            rs.process_fps = args.fps or 4
            process_file(str(real), rs, out=str(real_run), mode="demo", blur=True,
                         max_seconds=args.real_from + args.real_seconds + 2, write_video=True)
    else:
        print(f"нет {real} — реальный кусок пропущен (python scripts/fetch_videos.py)")
    out = build_reel(synth_run, real_run, Path(args.out), real_s=args.real_seconds, scene_fps=sc.fps,
                     real_from=args.real_from)
    print(f"Демо-ролик: {out}")
    return 0


def cmd_synth_week(args) -> int:
    from qonaq.synth.week import WeekParams, simulate

    site = _site(args)
    for path in args.params:
        p = WeekParams.load(path)
        out = Path(args.out) / p.name
        st = simulate(p, site, out)
        n = len(st.sessions())
        st.close()
        print(f"{p.name}: {p.days} дн., {p.tables} столов, посадок {n} -> {out / 'events.sqlite'}")
    return 0


def cmd_billiard(args) -> int:
    from qonaq.billiard.cli import cmd_billiard as run

    return run(args)


def cmd_reconcile(args) -> int:
    from qonaq.reconcile import cmd_reconcile as run

    return run(args)


HANDLERS = {
    "reconcile": cmd_reconcile,
    "billiard": cmd_billiard,
    "demo-reel": cmd_demo_reel,
    "synth-week": cmd_synth_week,
    "service": cmd_service,
    "setup": cmd_setup,
    "telegram": cmd_telegram,
    "health": cmd_health,
    "calibrate": cmd_calibrate,
    "record": cmd_record,
    "staged-check": cmd_staged,
    "synth": cmd_synth,
    "train": cmd_train,
    "data": cmd_data,
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
