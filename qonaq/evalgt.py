"""Измерение качества (SPEC §9a п.5): сравнение событий системы с ручной разметкой.

Формат разметки `data/gt/<video>.yaml`:
    events:
      - {t: 12.5, zone: T03, event: seated}

Совпадение = тот же тип и та же зона в окне ±tolerance (по умолчанию 10 с), один-к-одному.
Выдаёт precision / recall по типам событий и среднюю абсолютную ошибку времени.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CORE_TYPES = ("seated", "served", "staff_visit", "left", "cleaned", "walkout_before_service",
              "seated_at_dirty_table", "wait_start", "wait_end", "service_start", "service_end")


@dataclass(frozen=True)
class Ev:
    t: float
    zone: str
    type: str


def load_gt(path: str | Path) -> list[Ev]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = data.get("events", data) if isinstance(data, dict) else data
    return [Ev(float(i["t"]), str(i["zone"]), str(i["event"])) for i in (items or [])]


def load_system(run_dir: str | Path) -> list[Ev]:
    from qonaq.events.store import EventStore

    store = EventStore(Path(run_dir) / "events.sqlite")
    try:
        return [Ev(float(e["ts"]), e["zone_id"], e["type"]) for e in store.events() if e["type"] in CORE_TYPES]
    finally:
        store.close()


def match_events(gt: list[Ev], sys: list[Ev], tolerance_s: float = 10.0):
    """Жадное сопоставление один-к-одному по минимальной разнице времени."""
    pairs = sorted(
        ((abs(g.t - s.t), gi, si) for gi, g in enumerate(gt) for si, s in enumerate(sys)
         if g.type == s.type and g.zone == s.zone and abs(g.t - s.t) <= tolerance_s),
        key=lambda x: x[0],
    )
    used_g, used_s, matches = set(), set(), []
    for d, gi, si in pairs:
        if gi in used_g or si in used_s:
            continue
        used_g.add(gi)
        used_s.add(si)
        matches.append((gt[gi], sys[si], d))
    missed = [g for i, g in enumerate(gt) if i not in used_g]
    extra = [s for i, s in enumerate(sys) if i not in used_s]
    return matches, missed, extra


def evaluate(gt: list[Ev], sys: list[Ev], tolerance_s: float = 10.0) -> dict[str, Any]:
    matches, missed, extra = match_events(gt, sys, tolerance_s)
    types = sorted({e.type for e in gt} | {e.type for e in sys})

    def row(tp: int, fn: int, fp: int, errs: list[float]) -> dict[str, Any]:
        return {
            "gt": tp + fn, "system": tp + fp, "tp": tp, "fn": fn, "fp": fp,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "time_error_s": round(sum(errs) / len(errs), 2) if errs else None,
        }

    by_type = {}
    for t in types:
        errs = [d for g, _, d in matches if g.type == t]
        by_type[t] = row(len(errs), sum(1 for g in missed if g.type == t), sum(1 for s in extra if s.type == t), errs)
    return {
        "tolerance_s": tolerance_s,
        "by_type": by_type,
        "total": row(len(matches), len(missed), len(extra), [d for *_, d in matches]),
        "missed": [{"t": e.t, "zone": e.zone, "event": e.type} for e in missed],
        "extra": [{"t": round(e.t, 1), "zone": e.zone, "event": e.type} for e in extra],
    }


def format_table(res: dict[str, Any]) -> str:
    f = lambda v: "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| Событие | В разметке | У системы | Совпало | Precision | Recall | Ошибка времени, с |",
             "|---|---|---|---|---|---|---|"]
    for name, r in [*res["by_type"].items(), ("ВСЕГО", res["total"])]:
        lines.append(f"| {name} | {r['gt']} | {r['system']} | {r['tp']} | {f(r['precision'])} | {f(r['recall'])} | "
                     f"{f(r['time_error_s'])} |")
    if res["missed"]:
        lines.append("\nПропущено: " + ", ".join(f"{e['event']}@{e['zone']} t={e['t']:.0f}" for e in res["missed"]))
    if res["extra"]:
        lines.append("\nЛишние: " + ", ".join(f"{e['event']}@{e['zone']} t={e['t']:.0f}" for e in res["extra"]))
    return "\n".join(lines)


def load_alerts(run_dir: str | Path) -> list[tuple[float, str, str]]:
    from qonaq.events.store import EventStore

    store = EventStore(Path(run_dir) / "events.sqlite")
    try:
        return [(float(e["ts"]), e["zone_id"], str((e.get("payload") or {}).get("kind")))
                for e in store.events(types=["alert"])]
    finally:
        store.close()


def check_alerts(gt_path: str | Path, alerts: list[tuple[float, str, str]]) -> dict[str, Any] | None:
    """Синтетика (W4): какие алерты обязаны быть (alerts_expected) и каких быть не должно (alerts_forbidden)."""
    data = yaml.safe_load(Path(gt_path).read_text(encoding="utf-8")) or {}
    exp, forb = data.get("alerts_expected") or [], data.get("alerts_forbidden") or []
    if not exp and not forb:
        return None
    kinds = [k for _, _, k in alerts]
    return {"expected": {k: k in kinds for k in exp}, "forbidden_fired": {k: kinds.count(k) for k in forb},
            "all": [{"t": round(t, 1), "zone": z, "kind": k} for t, z, k in alerts],
            "ok": all(k in kinds for k in exp) and not any(k in kinds for k in forb)}


def run_eval(run_dir: str | Path, gt_path: str | Path, tolerance_s: float = 10.0) -> dict[str, Any]:
    res = evaluate(load_gt(gt_path), load_system(run_dir), tolerance_s)
    res["gt_file"] = str(gt_path)
    al = check_alerts(gt_path, load_alerts(run_dir))
    if al is not None:
        res["alerts"] = al
    out = Path(run_dir) / "eval.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = format_table(res)
    if al is not None:
        md += ("\n\nАлерты: " + ("OK" if al["ok"] else "НЕ OK") + f"; ожидались {al['expected']}, "
               f"запрещённые сработали {al['forbidden_fired']}; все: "
               + (", ".join(f"{a['kind']}@{a['zone']} t={a['t']:.0f}" for a in al["all"]) or "нет"))
    (Path(run_dir) / "eval.md").write_text(md + "\n", encoding="utf-8")
    return res
