"""Постановочный тест одной строкой (W5): зоны → пайплайн → eval → демо с блюром → «что поймано».

    python -m qonaq staged-check data/videos/staged.mp4 --gt data/gt/staged.yaml

Зоны: configs/zones/<имя>.yaml, если есть; иначе автопредложение по детекциям; если столов не нашлось —
одна зона по умолчанию (как в configs/zones/demo_webcam.yaml: левая часть кадра).
Пороги — configs/sites/staged.yaml (боевые, но без правила «сидели до старта»).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from qonaq.config import PROJECT_ROOT, SiteConfig, load_site
from qonaq.zones.models import Zone, ZoneSet, load_zones, save_zones

log = logging.getLogger("qonaq.staged")

EVENT_RU = {
    "seated": "сели", "served": "подошёл официант (первый раз)", "staff_visit": "визит персонала",
    "left": "ушли", "cleaned": "стол убран", "walkout_before_service": "ушли, не дождавшись",
    "seated_at_dirty_table": "сели за неубранный стол",
}


def default_zone(w: int, h: int, camera_id: str) -> ZoneSet:
    """Одна зона-стол: левая часть кадра (как demo_webcam.yaml, 1280×720 → текущий размер)."""
    k = (w / 1280.0, h / 720.0)
    poly = [[40 * k[0], 60 * k[1]], [760 * k[0], 60 * k[1]], [760 * k[0], 700 * k[1]], [40 * k[0], 700 * k[1]]]
    return ZoneSet(camera_id=camera_id, frame_size=(w, h),
                   zones=[Zone(id="T01", type="table", name="Стол (по умолчанию)", polygon=poly)])


def ensure_zones(video: Path, site: SiteConfig) -> tuple[Path, str]:
    path = PROJECT_ROOT / "configs" / "zones" / f"{video.stem}.yaml"
    if path.exists():
        return path, "из файла"
    from qonaq.detect.base import create_detector
    from qonaq.zones.suggest import suggest_zones

    how = "автопредложение"
    try:
        zs, frame, _ = suggest_zones(video, create_detector(site.detector), 20, camera_id=video.stem)
    except Exception as e:  # noqa: BLE001
        log.warning("автопредложение зон не удалось: %s", e)
        zs = None
    if zs is None or not zs.by_type("table"):
        cap = cv2.VideoCapture(str(video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        zs = default_zone(w or 1280, h or 720, video.stem)
        how = "одна зона по умолчанию (столы не найдены)"
    save_zones(zs, path)
    return path, how


def caught_summary(run_dir: Path) -> str:
    from qonaq.events.store import EventStore

    store = EventStore(run_dir / "events.sqlite")
    try:
        evs = store.events()
    finally:
        store.close()
    lines = ["Что поймано:"]
    for e in evs:
        if e["type"] in EVENT_RU:
            p = e["payload"] or {}
            extra = ""
            if e["type"] == "served" and p.get("response_time_s") is not None:
                extra = f" (ждали {p['response_time_s']:.0f} с)"
            if e["type"] == "cleaned" and p.get("cleanup_time_s") is not None:
                extra = f" (через {p['cleanup_time_s']:.0f} с после ухода, by={p.get('by')})"
            lines.append(f"  {_mmss(e['ts'])}  {e['zone_id']}: {EVENT_RU[e['type']]}{extra}")
        elif e["type"] == "alert":
            lines.append(f"  {_mmss(e['ts'])}  {e['zone_id']}: АЛЕРТ {e['payload'].get('kind')}")
    if len(lines) == 1:
        lines.append("  ничего — проверьте зоны (runs/zones/*.png) и что человек виден детектором")
    return "\n".join(lines)


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def staged_check(video: str | Path, gt: str | Path | None, site_path: str | None = None, overrides=None) -> int:
    from qonaq.evalgt import format_table, run_eval
    from qonaq.runner import process_file
    from qonaq.zones.preview import save_preview

    video = Path(video)
    if not video.exists():
        raise SystemExit(f"нет видео: {video} (сначала: python -m qonaq record --source webcam:0 --seconds 600 "
                         f"--out {video})")
    site = load_site(site_path or str(PROJECT_ROOT / "configs" / "sites" / "staged.yaml"))
    if overrides:
        overrides(site)
    zpath, how = ensure_zones(video, site)
    print(f"Зоны: {zpath} ({how})")
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ok, frame = cap.read()
    cap.release()
    if ok:
        png = Path(site.runs_dir) / "zones" / f"{video.stem}.png"
        save_preview(png, frame, load_zones(zpath), [], blur=True)
        print(f"Превью зон: {png}")
    run_dir = Path(site.runs_dir) / video.stem
    process_file(str(video), site, out=str(run_dir), mode="demo", blur=True, write_video=True, snapshots=4,
                 zones_path=zpath)
    print(f"Демо-ролик (с блюром): {run_dir / 'demo.mp4'}")
    print(caught_summary(run_dir))
    gt_path = Path(gt) if gt else PROJECT_ROOT / "data" / "gt" / f"{video.stem}.yaml"
    if gt_path.exists():
        res = run_eval(run_dir, gt_path)
        print(f"\nСверка с разметкой {gt_path} (±{res['tolerance_s']:.0f} с):\n")
        print(format_table(res))
    else:
        print(f"\nРазметки {gt_path} нет — сверка пропущена (шаблон: docs/STAGED_TEST.md)")
    try:
        from qonaq.metrics.report import generate_report

        out = generate_report(run_dir, site)
        print(f"Отчёт: {out['html']}")
    except Exception as e:  # noqa: BLE001
        log.warning("отчёт не построен: %s", e)
    return 0
