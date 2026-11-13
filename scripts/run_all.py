"""Полный прогон на всех видео из data/videos/: демо-ролик (с блюром), отчёт, сверка с разметкой.

Запуск:  python scripts/run_all.py            (качество: yolox_m + плитки, 4 кадр/с — медленно на CPU)
         python scripts/run_all.py --fast     (yolox_s, 5 кадр/с)
Результаты: runs/<имя видео>/{demo.mp4, events.sqlite, report.html, report.md, telegram.txt, eval.md, health.json}
Синтетика (data/videos/synth_*) здесь не гоняется — для неё `python -m qonaq synth`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def run(*args: str) -> int:
    print("\n>>> python -m qonaq", " ".join(args), flush=True)
    return subprocess.call([sys.executable, "-m", "qonaq", *args], cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="лёгкая модель (yolox_s, без плиток)")
    ap.add_argument("--only", nargs="*", help="обработать только эти видео (по имени без расширения)")
    args = ap.parse_args()
    model = ["--model", "yolox_s", "--tiles", "1", "--fps", "5"] if args.fast else \
            ["--model", "yolox_m", "--tiles", "2", "--fps", "4"]
    # synth_* — свои пороги и разметка: их гоняет `python -m qonaq synth`; staged* — `staged-check`
    videos = sorted(p for p in (ROOT / "data" / "videos").glob("*") if p.suffix.lower() in VIDEO_EXT
                    and not p.stem.startswith(("synth_", "staged")))
    if not videos:
        print("нет видео: сначала python scripts/fetch_videos.py")
        return 1
    failed = []
    for v in videos:
        if args.only and v.stem not in args.only:
            continue
        rel = str(v.relative_to(ROOT))
        if run("demo", rel, "--backend", "onnx", *model, "--snapshots", "6"):
            failed.append(v.stem)
            continue
        run("report", f"runs/{v.stem}")
        gt_dirs = (ROOT / "data" / "gt", ROOT / "tests" / "fixtures" / "gt")
        if any((d / f"{v.stem}.yaml").exists() for d in gt_dirs):
            run("eval", f"runs/{v.stem}")
    print("\nГотово." + (f" Ошибки: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
