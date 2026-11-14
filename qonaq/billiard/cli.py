"""CLI бильярдной: python -m qonaq billiard synth|eval (B2/B4)."""
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
