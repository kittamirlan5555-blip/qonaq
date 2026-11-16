"""Трассировка прогона для разбора ошибок: что машина стола видела на каждом кадре (без изменения логики).

    python scripts/trace_run.py --site configs/sites/hogs_outside.yaml --out runs/hogs/trace \
        --det-cache runs/hogs/detcache --reel 1200:1290 --reel 1560:1700

Пишет `trace.jsonl.gz`: по кадру — люди (id, рамка, роль, уверенность, зоны, стоит) и по каждому столу
наблюдение машины (гости / персонал / неизвестные / стоящие), «сидящие за столом» (occupants), визит в работе и
состояние. `--reel a:b` — дополнительно демо-кадры (с блюром лиц, как `run-file --mode demo`) на интервале секунд
вырезки → `reel.mp4`. Видео и кадры не коммитить (runs/ в .gitignore).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qonaq.cli import _site  # noqa: E402
from qonaq.logic.restaurant import TableMachine  # noqa: E402
from qonaq.pipeline import Pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det-cache")
    ap.add_argument("--roles-head")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--reel", action="append", default=[], help="интервал секунд вырезки a:b для ролика")
    for k in ("fps", "backend", "model", "tiles", "device", "conf"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    a.fps = float(a.fps) if a.fps else None
    site = _site(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reels = [tuple(map(float, r.split(":"))) for r in a.reel]

    per_frame: dict[str, dict] = {}
    orig = TableMachine.update

    def traced(self, obs):
        evs = orig(self, obs)
        ep = self._episode
        per_frame[self.zone_id] = {
            "g": sorted(obs.guest_tracks), "s": sorted(obs.staff_tracks), "u": sorted(obs.unknown_tracks),
            "st": None if obs.standing_tracks is None else sorted(obs.standing_tracks),
            "occ": sorted(self._occupants), "fill": sorted(getattr(self, "_fillers", set())),
            "ep": None if ep is None else [round(ep.start, 1), sorted(ep.seen)],
            "state": str(self.state), "ev": [e.type for e in evs],
        }
        return evs

    TableMachine.update = traced
    pipe = Pipeline(site, max_seconds=a.max_seconds)
    cam_id = site.cameras[0].id
    cam = pipe.cameras[cam_id]
    sink = demo = None
    if reels:
        from qonaq.render.annotate import DemoRenderer
        from qonaq.render.sink import VideoSink

        sink = VideoSink(out / "reel.mp4", fps=site.process_fps)
        demo = DemoRenderer(cam.zones, site, blur=True)
    f = gzip.open(out / "trace.jsonl.gz", "wt", encoding="utf-8")

    def on_result(r) -> None:
        rec = {"t": round(r.ts, 2), "p": [[p.track_id, [round(v) for v in p.xyxy], p.role[:1], round(p.confidence, 2),
                                         p.zones, int(p.standing)] for p in r.people],
               "z": dict(per_frame)}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        per_frame.clear()
        if demo is not None:
            img = demo.render(r)  # рендер на каждом кадре: у демо своё состояние (таймеры, лента событий)
            if any(x <= r.ts <= y for x, y in reels):
                sink.write(img)

    pipe.run(on_result=on_result, progress_every_s=30)
    pipe.finish()
    f.close()
    if sink is not None:
        sink.close()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
