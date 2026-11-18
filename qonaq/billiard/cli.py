"""CLI бильярдной: python -m qonaq billiard synth|eval|ledger (B2/B4, этап 3 — учёт столов)."""
from __future__ import annotations

import json
from pathlib import Path


def add_parser(sub) -> None:
    bp = sub.add_parser("billiard", help="бильярдная: synth — синтетика + прогон + оценка; eval — интервалы игры")
    b = bp.add_subparsers(dest="b_cmd", required=True)
    p = b.add_parser("synth", help="сценарий YAML -> ролик + зоны + правда; --run — прогон и оценка")
    p.add_argument("scenario", nargs="+")
    p.add_argument("--out", default="data/videos", help="куда положить ролик и разметку")
    p.add_argument("--run", action="store_true", help="прогнать профиль billiard и сравнить с правдой")
    p.add_argument("--demo", action="store_true", help="с --run: записать demo.mp4 (панель)")
    p.add_argument("--reuse", action="store_true", help="не рендерить заново, если ролик уже есть")
    p = b.add_parser("eval", help="events.sqlite против разметки (IoU по времени, ошибка в минутах)")
    p.add_argument("--events", required=True)
    p.add_argument("--gt", required=True)
    p = b.add_parser("ledger", help="учёт столов по видео: часы игры, сумма по прайсу, загрузка — по рабочим дням")
    p.add_argument("--events", required=True, help="events.sqlite (live или прогон файла)")
    p.add_argument("--site", help="YAML объекта: тарифы billiard.*, часы работы, часовой пояс, столы")
    p.add_argument("--date", action="append", help="рабочий день ГГГГ-ММ-ДД (можно несколько); нет — все дни журнала")
    p.add_argument("--day-start", type=int, help="час начала рабочего дня (по умолчанию из объекта, 6)")
    p.add_argument("--video-start", help="журнал прогона файла: местное время начала записи ГГГГ-ММ-ДД ЧЧ:ММ")
    p.add_argument("--timezone", help="часовой пояс, если нет --site (по умолчанию Asia/Almaty)")
    p.add_argument("--title", default="", help="подпись в отчёте (название клуба)")
    p.add_argument("--out", help="папка отчёта (по умолчанию <папка журнала>/ledger)")
    p.add_argument("--send", action="store_true", help="отправить сводку последнего дня в чат заведения")


def run_video(video: Path, zones: Path, out: Path, demo: bool = False) -> Path:
    from qonaq.config import SiteConfig
    from qonaq.detect.base import FakeDetector
    from qonaq.runner import process_file

    site = SiteConfig(profile="billiard", process_fps=2.0)
    site.privacy.blur_faces = True
    # Детектор людей бильярдному столу не нужен (решение — по сукну); на синтетике людей-кружков он и не
    # увидит. На реальном видео прогон через `python -m qonaq run-file --site <billiard.yaml>`.
    return process_file(video, site, out=out, mode="demo" if demo else "tracks", write_video=demo,
                        detector=FakeDetector(), zones_path=zones)


def cmd_billiard(args) -> int:
    from qonaq.billiard.evaluate import evaluate, format_table, load_gt, sessions_from_events
    from qonaq.events.store import EventStore

    if args.b_cmd == "ledger":
        return cmd_ledger(args)
    if args.b_cmd == "eval":
        st = EventStore(args.events)
        evs = st.events(types=["game_started", "game_ended"])
        res = evaluate(sessions_from_events(evs), load_gt(args.gt))
        print(format_table(res))
        return 0
    from qonaq.billiard.synth import build

    rc = 0
    for scen in args.scenario:
        import yaml

        name = "synth_" + yaml.safe_load(Path(scen).read_text(encoding="utf-8"))["name"]
        out = Path(args.out)
        video, zones, gt = out / f"{name}.mp4", out / f"{name}.yaml", out / f"{name}.gt.yaml"
        if not (args.reuse and video.exists() and gt.exists()):
            info = build(Path(scen), out)
            print(f"{scen}: {info['frames']} кадров -> {info['video']}")
        if not args.run:
            continue
        run_dir = run_video(video, zones, Path("runs") / name, demo=args.demo)
        st = EventStore(run_dir / "events.sqlite")
        res = evaluate(sessions_from_events(st.events(types=["game_started", "game_ended"])), load_gt(gt))
        (run_dir / "billiard_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n{name}\n" + format_table(res))
        rc |= int(res["missed"] > 0 or res["false"] > 0)
    return rc


def cmd_ledger(args) -> int:
    from datetime import date

    from qonaq.billiard.ledger import LedgerSettings, from_store, telegram_text, write_outputs
    from qonaq.config import SiteConfig, load_site
    from qonaq.events.store import EventStore
    from qonaq.reconcile import local_tz, parse_dt

    site = load_site(args.site) if args.site else SiteConfig(profile="billiard")
    if args.timezone:
        site.business.timezone = args.timezone
    tz = local_tz(site.business.timezone)
    cfg = LedgerSettings.from_site(site)
    if args.day_start is not None:
        cfg.day_start_h = args.day_start
    tables: set[str] = set()
    if args.site:
        from qonaq.pipeline import load_camera_zones

        for cam in site.cameras:
            for z in load_camera_zones(cam).by_type("billiard_table"):
                tables.add(z.id)
                if z.name:
                    cfg.titles[z.id] = z.name
    store = EventStore(args.events)
    offset = 0.0
    if store.meta().get("time_base", "video") != "epoch":
        if not args.video_start:
            print("Журнал прогона файла (время от начала видео): укажите --video-start «ГГГГ-ММ-ДД ЧЧ:ММ» — "
                  "местное время начала записи (видно в OSD регистратора).")
            store.close()
            return 2
        offset = parse_dt(args.video_start, None, tz).timestamp()
    days = from_store(store, cfg, tz, offset, tables, [date.fromisoformat(d) for d in args.date or []] or None)
    store.close()
    out = Path(args.out) if args.out else Path(args.events).parent / "ledger"
    paths = write_outputs(days, out, tz, args.title or site.name)
    for d in days:
        print(telegram_text(d, args.title or site.name))
        print()
    if not days:
        print("В журнале нет игр за выбранные дни.")
    print(f"-> {paths['html']}")
    if args.send and days:
        from qonaq.alerts.telegram import sender_from_env

        s = sender_from_env(site.alerts.venue_chat_env)
        if s is None:
            print(f"(не отправлено: нет TELEGRAM_BOT_TOKEN / {site.alerts.venue_chat_env})")
        else:
            s.send(telegram_text(days[-1], args.title or site.name))
    return 0
