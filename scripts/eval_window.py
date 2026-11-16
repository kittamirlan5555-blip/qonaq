"""Сверка прогона с разметкой по ОКНУ записи (независимая проверка на чужой камере, EVAL «Hog's Breath»).

    python scripts/eval_window.py runs/hogs/nocalib data/gt/hogs_outside_1116.yaml --offset 900

Отличия от `qonaq eval` (qonaq/evalgt.py):
- время разметки — от начала исходного файла, прогон — по вырезке с `--offset` секунд;
- оцениваются только события внутри `window` разметки (до него — прогрев: столы, занятые до окна, система уже видит);
- `unscored` — зоны/типы/интервалы, где разметка по кадрам ненадёжна (сторона подхода за краем кадра):
  события системы там не считаются ни лишними, ни совпавшими;
- ошибка по времени — со знаком (система − разметка), среднее по модулю и максимум;
- `alerts_forbidden` — по зонам; `not_events` — подписи к лишним событиям, попавшим в известные «ловушки».
Совпадение — тот же тип и зона в пределах ±tolerance (10 с), один-к-одному (как в qonaq/evalgt.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qonaq.evalgt import CORE_TYPES, Ev, match_events  # noqa: E402
from qonaq.events.store import EventStore  # noqa: E402

TYPES = ("seated", "served", "staff_visit", "left", "cleaned")


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(round(t % 60)):02d}" if t % 60 < 59.5 else f"{int(t // 60) + 1}:00"


def load_run(run_dir: Path, offset: float) -> tuple[list[Ev], list[dict], list[dict]]:
    st = EventStore(run_dir / "events.sqlite")
    try:
        evs = st.events()
    finally:
        st.close()
    core = [Ev(float(e["ts"]) + offset, e["zone_id"], e["type"]) for e in evs if e["type"] in CORE_TYPES]
    alerts = [{"t": float(e["ts"]) + offset, "zone": e["zone_id"], "kind": (e.get("payload") or {}).get("kind"),
               "payload": e.get("payload")} for e in evs if e["type"] == "alert"]
    raw = [{"t": float(e["ts"]) + offset, "zone": e["zone_id"], "type": e["type"], "payload": e.get("payload")}
           for e in evs]
    return core, alerts, raw


def unscored(gt: dict, e: Ev) -> bool:
    for u in gt.get("unscored") or []:
        a, b = u.get("span", [-1e18, 1e18])
        if u["zone"] == e.zone and e.type in u["events"] and a <= e.t <= b:
            return True
    return False


def trap(gt: dict, e: Ev, pad: float = 15.0) -> str | None:
    for n in gt.get("not_events") or []:
        a, b = n["span"]
        if n["zone"] == e.zone and a - pad <= e.t <= b + pad:
            return n["note"]
    return None


def evaluate(run_dir: Path, gt_path: Path, offset: float, tol: float = 10.0) -> dict:
    gt = yaml.safe_load(gt_path.read_text(encoding="utf-8"))
    w0, w1 = gt["window"]
    g = [Ev(float(i["t"]), i["zone"], i["event"]) for i in gt["events"] if w0 <= float(i["t"]) <= w1]
    notes = {(float(i["t"]), i["zone"], i["event"]): i.get("note", "") for i in gt["events"]}
    sys_all, alerts, raw = load_run(run_dir, offset)
    # события системы у границ окна: ±tol, чтобы событие на 17:58 не стало лишним к разметке на 18:03
    s = [e for e in sys_all if e.type in TYPES and w0 - tol <= e.t <= w1 + tol and not unscored(gt, e)]
    s_in = [e for e in s if w0 <= e.t <= w1 or any(abs(e.t - x.t) <= tol and e.zone == x.zone and e.type == x.type
                                                     for x in g)]
    matches, missed, extra = match_events(g, s_in, tol)
    rows = {}
    for t in TYPES:
        m = [(gg, ss, ss.t - gg.t) for gg, ss, _ in matches if gg.type == t]
        rows[t] = {"gt": sum(1 for x in g if x.type == t), "system": sum(1 for x in s_in if x.type == t),
                   "tp": len(m), "fn": sum(1 for x in missed if x.type == t),
                   "fp": sum(1 for x in extra if x.type == t),
                   "dt": [round(d, 1) for *_, d in m],
                   "mae": round(sum(abs(d) for *_, d in m) / len(m), 1) if m else None,
                   "max": round(max(abs(d) for *_, d in m), 1) if m else None}
    forb = gt.get("alerts_forbidden") or []
    al_in = [a for a in alerts if w0 <= a["t"] <= w1]
    fired_forbidden = [a for a in al_in if any(f["zone"] == a["zone"] and f["kind"] == a["kind"] for f in forb)]
    return {
        "window": [w0, w1], "offset": offset, "tolerance_s": tol, "rows": rows,
        "matches": [{"zone": a.zone, "event": a.type, "gt": a.t, "sys": round(b.t, 1), "dt": round(b.t - a.t, 1)}
                    for a, b, _ in sorted(matches, key=lambda m: m[0].t)],
        "missed": [{"zone": e.zone, "event": e.type, "t": e.t, "note": notes.get((e.t, e.zone, e.type), "")}
                   for e in sorted(missed, key=lambda e: e.t)],
        "extra": [{"zone": e.zone, "event": e.type, "t": round(e.t, 1), "trap": trap(gt, e)}
                  for e in sorted(extra, key=lambda e: e.t)],
        "unscored_system": [{"zone": e.zone, "event": e.type, "t": round(e.t, 1)}
                            for e in sys_all if w0 <= e.t <= w1 and e.type in TYPES and unscored(gt, e)],
        "alerts": [{"zone": a["zone"], "kind": a["kind"], "t": round(a["t"], 1)} for a in al_in],
        "alerts_forbidden_fired": [{"zone": a["zone"], "kind": a["kind"], "t": round(a["t"], 1)}
                                   for a in fired_forbidden],
        "all_events": [{"t": round(r["t"], 1), "zone": r["zone"], "type": r["type"],
                        **({"conf": r["payload"].get("confidence")} if r["payload"] else {})}
                       for r in raw if r["type"] in TYPES or r["type"] == "alert"],
    }


def to_md(res: dict, title: str) -> str:
    f = lambda v: "—" if v is None else str(v)  # noqa: E731
    out = [f"### {title}", "",
           "| Событие | В разметке | Поймано | Пропущено | Лишних | Ошибка времени, с (среднее / макс) |",
           "|---|---|---|---|---|---|"]
    tot = {"gt": 0, "tp": 0, "fn": 0, "fp": 0}
    for t, r in res["rows"].items():
        out.append(f"| {t} | {r['gt']} | {r['tp']} | {r['fn']} | {r['fp']} | {f(r['mae'])} / {f(r['max'])} |")
        for k in tot:
            tot[k] += r[k]
    out.append(f"| **всего** | {tot['gt']} | {tot['tp']} | {tot['fn']} | {tot['fp']} | |")
    out += ["", "Совпало (разметка → система, Δ = система − разметка):"]
    out += [f"- {m['zone']} {m['event']}: {mmss(m['gt'])} → {mmss(m['sys'])} (Δ {m['dt']:+.0f} с)"
            for m in res["matches"]]
    if res["missed"]:
        out += ["", "Пропущено:"]
        out += [f"- {m['zone']} {m['event']} {mmss(m['t'])} — {m['note']}" for m in res["missed"]]
    if res["extra"]:
        out += ["", "Лишние:"]
        out += [f"- {e['zone']} {e['event']} {mmss(e['t'])}" + (f" — ловушка: {e['trap']}" if e["trap"] else "")
                for e in res["extra"]]
    if res["unscored_system"]:
        out += ["", "Не оценивались (сторона подхода за краем кадра): "
                + ", ".join(f"{e['zone']} {e['event']} {mmss(e['t'])}" for e in res["unscored_system"])]
    fmt = lambda xs: ", ".join(f"{a['kind']}@{a['zone']} {mmss(a['t'])}" for a in xs) or "нет"  # noqa: E731
    out += ["", f"Алерты в окне: {fmt(res['alerts'])}; запрещённые сработали: {fmt(res['alerts_forbidden_fired'])}"]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("gt")
    ap.add_argument("--offset", type=float, default=0.0, help="начало вырезки в секундах исходного файла")
    ap.add_argument("--tol", type=float, default=10.0)
    ap.add_argument("--title", default="Сверка")
    a = ap.parse_args()
    res = evaluate(Path(a.run_dir), Path(a.gt), a.offset, a.tol)
    (Path(a.run_dir) / "eval_window.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_md(res, a.title)
    (Path(a.run_dir) / "eval_window.md").write_text(md + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
