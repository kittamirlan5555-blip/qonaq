"""Оценка интервалов игры (B4): сессии по видео против разметки — IoU по времени, ошибка в минутах."""
from __future__ import annotations

from pathlib import Path
from typing import Any

Interval = tuple[float, float]


def merge(iv: list[Interval]) -> list[Interval]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total(iv: list[Interval]) -> float:
    return sum(b - a for a, b in merge(iv))


def intersect(x: list[Interval], y: list[Interval]) -> float:
    s = 0.0
    for a, b in merge(x):
        for c, d in merge(y):
            s += max(0.0, min(b, d) - max(a, c))
    return s


def sessions_from_events(events: list[dict[str, Any]], end_ts: float | None = None) -> dict[str, list[Interval]]:
    """game_started/game_ended → интервалы по столам. Незакрытая сессия — до end_ts (или последнего события)."""
    out: dict[str, list[Interval]] = {}
    open_: dict[tuple[str, str], float] = {}
    open_det: dict[tuple[str, str], float] = {}
    last = max((e["ts"] for e in events), default=0.0)
    for e in sorted(events, key=lambda e: e["ts"]):
        key = (e.get("camera_id") or "", e["zone_id"])
        p = e.get("payload") or {}
        if e["type"] == "game_started":
            if key in open_:  # прошлая игра не закрыта (старый журнал, сбой) — до её последнего подтверждения
                a = open_.pop(key)
                out.setdefault(e["zone_id"], []).append((a, max(a, open_det.get(key, a))))
            open_[key] = float(p.get("start", e["ts"]))
            open_det[key] = float(p.get("detected_ts", e["ts"]))
        elif e["type"] == "game_ended":
            open_.pop(key, None)
            out.setdefault(e["zone_id"], []).append((float(p["start"]), float(p["end"])))
    for (_, zid), a in open_.items():
        out.setdefault(zid, []).append((a, end_ts if end_ts is not None else last))
    return out


def evaluate(pred: dict[str, list[Interval]], gt: dict[str, list[Interval]]) -> dict[str, Any]:
    """IoU по времени (по столу и в сумме), минуты игры (правда / по видео), ложные и пропущенные сессии,
    ошибки начала/конца у сопоставленных сессий (по максимальному перекрытию)."""
    rows = []
    inter_all = union_all = 0.0
    for tid in sorted(set(pred) | set(gt)):
        p, g = merge(pred.get(tid, [])), merge(gt.get(tid, []))
        inter = intersect(p, g)
        union = total(p) + total(g) - inter
        inter_all += inter
        union_all += union
        matched, starts, ends = set(), [], []
        for a, b in g:
            best = max(p, key=lambda q, a=a, b=b: max(0.0, min(b, q[1]) - max(a, q[0])), default=None)
            if best is not None and min(b, best[1]) - max(a, best[0]) > 0:
                matched.add(best)
                starts.append(best[0] - a)
                ends.append(best[1] - b)
        missed = sum(1 for a, b in g if intersect([(a, b)], p) <= 0)
        false = sum(1 for q in p if intersect([q], g) <= 0)
        rows.append({"table": tid, "gt_sessions": len(g), "pred_sessions": len(p), "missed": missed, "false": false,
                     "gt_min": round(total(g) / 60, 2), "pred_min": round(total(p) / 60, 2),
                     "err_min": round((total(p) - total(g)) / 60, 2),
                     "iou": round(inter / union, 3) if union > 0 else 1.0,
                     "start_err_s": [round(x, 1) for x in starts], "end_err_s": [round(x, 1) for x in ends]})
    return {"tables": rows, "iou": round(inter_all / union_all, 3) if union_all > 0 else 1.0,
            "gt_min": round(sum(r["gt_min"] for r in rows), 2), "pred_min": round(sum(r["pred_min"] for r in rows), 2),
            "missed": sum(r["missed"] for r in rows), "false": sum(r["false"] for r in rows)}


def load_gt(path: str | Path, session_gap_s: float | None = None) -> dict[str, list[Interval]]:
    """Разметка: интервалы игры по столам. Игры одного стола с перерывом короче session_gap_s (по умолчанию
    из разметки или 600 с = P) — одна сессия: так её считает и учёт (компания взяла стол и сделала перерыв)."""
    import yaml

    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gap = float(session_gap_s if session_gap_s is not None else d.get("session_gap_s", 600.0))
    out: dict[str, list[Interval]] = {}
    for tid, v in (d.get("tables") or {}).items():
        merged: list[list[float]] = []
        for a, b in sorted(tuple(x) for x in v.get("games", [])):
            if merged and a - merged[-1][1] < gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out[tid] = [(a, b) for a, b in merged]
    return out


def format_table(res: dict[str, Any]) -> str:
    lines = ["| Стол | Сессий (правда/видео) | Пропущено | Ложных | Минут (правда/видео) | Ошибка, мин | IoU |"
             " Ошибка начала, с | Ошибка конца, с |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res["tables"]:
        lines.append(f"| {r['table']} | {r['gt_sessions']}/{r['pred_sessions']} | {r['missed']} | {r['false']} | "
                     f"{r['gt_min']}/{r['pred_min']} | {r['err_min']:+} | {r['iou']} | "
                     f"{', '.join(f'{x:+.0f}' for x in r['start_err_s']) or '—'} | "
                     f"{', '.join(f'{x:+.0f}' for x in r['end_err_s']) or '—'} |")
    lines.append(f"| **Итого** | | {res['missed']} | {res['false']} | {res['gt_min']}/{res['pred_min']} | "
                 f"{res['pred_min'] - res['gt_min']:+.2f} | **{res['iou']}** | | |")
    return "\n".join(lines)
