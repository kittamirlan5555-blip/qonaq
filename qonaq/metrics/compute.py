"""Метрики смены и ROI — считаются ТОЛЬКО из событий/сессий в SQLite, не из видео (SPEC §7)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Any

from qonaq.config import SiteConfig
from qonaq.events.store import EventStore

ALMATY = timezone(timedelta(hours=5))  # Asia/Almaty, UTC+5 (единая зона РК с 2024 г.)
Interval = tuple[float, float]


def percentile(values: list[float], q: float) -> float | None:
    """Линейная интерполяция, q в [0..100]."""
    if not values:
        return None
    v = sorted(values)
    if len(v) == 1:
        return float(v[0])
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (k - lo))


def stats(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {"n": len(vals), "mean": round(mean(vals), 1), "median": round(median(vals), 1),
            "p90": round(percentile(vals, 90), 1), "max": round(max(vals), 1)}


def merge_intervals(iv: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for a, b in sorted(i for i in iv if i[1] > i[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def overlap_len(a: Interval, merged: list[Interval]) -> float:
    return sum(max(0.0, min(a[1], b[1]) - max(a[0], b[0])) for b in merged)


def step_intervals(points: list[tuple[float, float]], t_end: float, predicate) -> list[Interval]:
    """Ступенчатая функция (ts, value) -> интервалы, где predicate(value) истинно."""
    out = []
    for i, (ts, v) in enumerate(points):
        nxt = points[i + 1][0] if i + 1 < len(points) else t_end
        if predicate(v) and nxt > ts:
            out.append((ts, nxt))
    return merge_intervals(out)


def occupancy_steps(intervals: list[Interval], t0: float) -> list[tuple[float, float]]:
    """Интервалы занятости столов -> ступенчатая функция «сколько столов занято»."""
    marks = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    steps, n = [(t0, 0.0)], 0
    for ts, d in marks:
        n += d
        if steps and steps[-1][0] == ts:
            steps[-1] = (ts, float(n))
        else:
            steps.append((ts, float(n)))
    return steps


def fmt_ts(ts: float, epoch: bool, tz=None) -> str:
    if epoch:
        return datetime.fromtimestamp(ts, tz or ALMATY).strftime("%H:%M")
    ts = max(0, int(ts))
    return f"{ts // 3600}:{ts % 3600 // 60:02d}:{ts % 60:02d}" if ts >= 3600 else f"{ts // 60:02d}:{ts % 60:02d}"


def choose_bucket(duration: float, epoch: bool) -> float:
    if epoch or duration >= 3 * 3600:
        return 3600.0
    for b in (10, 30, 60, 300, 600, 1800):
        if duration / b <= 12:
            return float(b)
    return 3600.0


def compute_metrics(store: EventStore, site: SiteConfig | None = None, zones: list[str] | None = None,
                    t_end: float | None = None) -> dict[str, Any]:
    site = site or SiteConfig()
    biz, al = site.business, site.alerts
    meta = store.meta()
    events = store.events()
    sessions = store.sessions("table")
    epoch = meta.get("time_base") == "epoch"
    from qonaq.config import business_tz

    tz = business_tz(biz.timezone)
    all_ts = [e["ts"] for e in events] + [s["seated_ts"] for s in sessions]
    t0 = float(meta.get("t_start", min(all_ts) if (epoch and all_ts) else 0.0))
    if t_end is None:
        t_end = float(meta.get("t_end", 0.0)) or (t0 + float(meta.get("scene_s", 0.0)))
    last = max([e["payload"].get("detected_ts", e["ts"]) for e in events] + [t0])
    t_end = max(t_end, last)
    duration = max(t_end - t0, 1e-9)

    table_ids = sorted(set(zones or []) | {s["zone_id"] for s in sessions})
    n_tables = int(meta.get("tables") or 0) or len(table_ids)
    n_tables = max(n_tables, len(table_ids))

    measured = [s for s in sessions if not s["preexisting"]]
    # ---------------- время до первого подхода
    resp = [s["response_time_s"] for s in measured if s["response_time_s"] is not None]
    response = stats(resp)
    served = [s for s in measured if s["served_ts"] is not None]
    still_waiting = [s for s in measured if s["served_ts"] is None and s["outcome"] == "open"]
    # Столы, к которым так и не подошли (ждут до сих пор или ушли, не дождавшись), но ждавшие дольше порога,
    # обязаны попасть в «долю ожиданий дольше порога»: иначе самые плохие случаи выпадают из статистики
    # (ошибка выжившего) и отчёт показывает «0%» при трёх горящих алертах.
    unserved_over = [s for s in measured if s["served_ts"] is None and (
        (s["outcome"] == "open" and t_end - s["seated_ts"] > al.no_waiter_s)
        or (s["outcome"] == "walkout" and (s["visit_duration_s"] or 0) > al.no_waiter_s))]
    over_n = sum(1 for r in resp if r > al.no_waiter_s) + len(unserved_over)
    denom = len(resp) + len(unserved_over)
    response["share_over_threshold"] = round(over_n / denom, 3) if denom else None
    response["unserved_over_threshold"] = len(unserved_over)
    response["threshold_s"] = al.no_waiter_s
    response["low_confidence_share"] = (
        round(sum(1 for s in served if s["low_confidence"]) / len(served), 3) if served else None)

    # ---------------- уход до обслуживания
    walkouts = [s for s in sessions if s["outcome"] == "walkout"]

    # ---------------- грязные столы
    by_zone: dict[str, list[dict]] = {}
    for s in sessions:
        by_zone.setdefault(s["zone_id"], []).append(s)
    dirty_intervals: list[tuple[str, float, float, bool]] = []  # zone, from, to, closed
    for zid, ss in by_zone.items():
        ss.sort(key=lambda s: s["seated_ts"])
        for i, s in enumerate(ss):
            if s["outcome"] not in ("left", "completed") or s["left_ts"] is None:
                continue
            if s["cleaned_ts"] is not None:
                dirty_intervals.append((zid, s["left_ts"], s["cleaned_ts"], True))
            elif i + 1 < len(ss):
                dirty_intervals.append((zid, s["left_ts"], ss[i + 1]["seated_ts"], True))
            else:
                dirty_intervals.append((zid, s["left_ts"], t_end, False))
    dirty = stats([b - a for _, a, b, _ in dirty_intervals])
    dirty["open"] = sum(1 for d in dirty_intervals if not d[3])
    dirty["total_s"] = round(sum(b - a for _, a, b, _ in dirty_intervals), 1)
    dirty["seated_at_dirty"] = sum(1 for s in sessions if s["dirty_seat"])

    # ---------------- занятость, пик
    occ_intervals = {zid: merge_intervals([(s["seated_ts"], s["left_ts"] if s["left_ts"] is not None else t_end)
                                           for s in ss]) for zid, ss in by_zone.items()}
    flat = [iv for ivs in occ_intervals.values() for iv in ivs]
    occ = occupancy_steps(flat, t0)
    peak_tables = biz.peak_occupancy * n_tables if n_tables else float("inf")
    peak_by_load = step_intervals(occ, t_end, lambda v: n_tables > 0 and v >= peak_tables - 1e-9)
    queue_pts = [(e["ts"], float(e["payload"].get("count", 0))) for e in events if e["type"] == "queue"]
    peak_by_queue = step_intervals(queue_pts, t_end, lambda v: v > 0)
    peak = merge_intervals(peak_by_load + peak_by_queue)
    peak_s = sum(b - a for a, b in peak)
    occupied_table_s = sum(b - a for a, b in flat)
    avg_load = occupied_table_s / (n_tables * duration) if n_tables else 0.0
    dirty_merged = {zid: merge_intervals([(a, b) for z, a, b, _ in dirty_intervals if z == zid])
                    for zid in table_ids}
    free_in_peak = 0.0
    for zid in table_ids:
        busy = merge_intervals(occ_intervals.get(zid, []) + dirty_merged.get(zid, []))
        free_in_peak += peak_s - sum(overlap_len(iv, peak) for iv in busy)
    free_in_peak += peak_s * max(0, n_tables - len(table_ids))

    # ---------------- визиты, оборачиваемость
    finished = [s for s in measured if s["outcome"] in ("left", "completed") and s["visit_duration_s"] is not None]
    visit = stats([s["visit_duration_s"] for s in finished])
    visits_per_session = stats([s["staff_visits"] for s in sessions if s["outcome"] != "walkout"])
    seatings = len(sessions)
    turnover = {
        "seatings": seatings, "new_seatings": len(measured),
        "per_table": round(seatings / n_tables, 2) if n_tables else None,
        "per_table_hour": round(len(measured) / n_tables / (duration / 3600), 2) if n_tables else None,
    }

    # ---------------- алерты
    alerts: dict[str, int] = {}
    for e in events:
        if e["type"] == "alert":
            k = e["payload"].get("kind", "?")
            alerts[k] = alerts.get(k, 0) + 1

    # ---------------- ROI (SPEC §7): только лишний простой грязного стола в пиковое время
    avg_visit_s = visit["mean"] if visit["n"] >= 5 else biz.default_visit_min * 60
    excess = 0.0
    for _, a, b, _ in dirty_intervals:
        excess += max(0.0, overlap_len((a, b), peak) - biz.target_cleanup_s)
    lost_seatings = excess / avg_visit_s if avg_visit_s else 0.0
    roi = {
        "peak_s": round(peak_s, 1), "peak_share": round(peak_s / duration, 3),
        "excess_dirty_in_peak_s": round(excess, 1),
        "avg_visit_s": round(avg_visit_s, 1), "avg_visit_measured": visit["n"] >= 5,
        "lost_seatings": round(lost_seatings, 2),
        "loss_kzt": round(lost_seatings * biz.avg_check_kzt),
        "walkouts": len(walkouts), "walkout_loss_kzt": round(len(walkouts) * biz.avg_check_kzt),
        "avg_check_kzt": biz.avg_check_kzt, "target_cleanup_s": biz.target_cleanup_s,
        "free_table_s_in_peak": round(free_in_peak, 1),
    }

    # ---------------- по столам
    per_table = []
    for zid in table_ids:
        ss = by_zone.get(zid, [])
        r = [s["response_time_s"] for s in ss if s["response_time_s"] is not None and not s["preexisting"]]
        d = [(b - a) for z, a, b, _ in dirty_intervals if z == zid]
        n_alerts = sum(1 for e in events if e["type"] == "alert" and e["zone_id"] == zid)
        n_walk = sum(1 for s in ss if s["outcome"] == "walkout")
        occ_s = sum(b - a for a, b in occ_intervals.get(zid, []))
        over = sum(1 for x in r if x > al.no_waiter_s)
        per_table.append({
            "zone_id": zid, "sessions": len(ss), "avg_response_s": round(mean(r), 1) if r else None,
            "max_response_s": round(max(r), 1) if r else None, "dirty_total_s": round(sum(d), 1),
            "alerts": n_alerts, "walkouts": n_walk, "load": round(occ_s / duration, 3),
            # балл «проблемности»: алерты, уходы, долгие ожидания и минуты грязного стола
            "problem_score": round(2 * n_alerts + 3 * n_walk + 2 * over + sum(d) / 300.0, 2),
        })
    problems = sorted((p for p in per_table if p["problem_score"] > 0), key=lambda p: -p["problem_score"])

    # ---------------- загрузка по времени (heatmap: интервал x стол)
    bucket = choose_bucket(duration, epoch)
    if epoch:
        start = (t0 // 3600) * 3600
    else:
        start = t0
    edges = []
    t = start
    while t < t_end and len(edges) < 200:
        edges.append((max(t, t0), min(t + bucket, t_end)))
        t += bucket
    heat = {"bucket_s": bucket, "labels": [fmt_ts(a, epoch, tz) for a, _ in edges], "rows": []}
    for zid in table_ids:
        row = []
        for a, b in edges:
            row.append(round(overlap_len((a, b), occ_intervals.get(zid, [])) / max(b - a, 1e-9), 2))
        heat["rows"].append({"zone_id": zid, "values": row})
    heat["total"] = [
        round(sum(r["values"][i] for r in heat["rows"]) / n_tables, 2) if n_tables else 0.0
        for i in range(len(edges))
    ]
    queue = {"max": int(max((v for _, v in queue_pts), default=0)),
             "time_with_queue_s": round(sum(b - a for a, b in peak_by_queue), 1)} if queue_pts else None

    return {
        "site": {"id": site.site_id, "name": site.name, "profile": site.profile},
        "period": {"t0": t0, "t_end": t_end, "duration_s": round(duration, 1), "epoch": epoch,
                   "from": fmt_ts(t0, epoch, tz), "to": fmt_ts(t_end, epoch, tz)},
        "tables": {"total": n_tables, "with_activity": len(table_ids), "avg_load": round(avg_load, 3)},
        "response": response,
        "still_waiting": len(still_waiting),
        "walkouts": {"n": len(walkouts), **{k: v for k, v in stats(
            [s["visit_duration_s"] for s in walkouts]).items() if k in ("mean", "max")}},
        "dirty": dirty,
        "visit": visit,
        "visits_per_session": visits_per_session,
        "turnover": turnover,
        "alerts": alerts,
        "roi": roi,
        "per_table": per_table,
        "problem_tables": problems[:5],
        "heatmap": heat,
        "queue": queue,
        "quality": {
            "preexisting_sessions": sum(1 for s in sessions if s["preexisting"]),
            "low_confidence_sessions": sum(1 for s in sessions if s["low_confidence"]),
            "sessions": len(sessions), "events": len(events),
        },
        "clinic": clinic_metrics(store, al.long_wait_s) if site.profile == "clinic" else None,
    }


def clinic_metrics(store: EventStore, long_wait_s: float = 900.0) -> dict[str, Any]:
    waits = store.sessions("wait")
    services = store.sessions("service")
    done = [w for w in waits if w["outcome"] != "open" and w["visit_duration_s"] is not None]
    w = stats([x["visit_duration_s"] for x in done])
    long_n = sum(1 for x in done if x["visit_duration_s"] > long_wait_s)
    w["long_share"] = round(long_n / len(done), 3) if done else None
    return {
        "waits": w,
        "served": sum(1 for x in waits if x["outcome"] == "served"),
        "left_without_service": sum(1 for x in waits if x["outcome"] == "left"),
        "open": sum(1 for x in waits if x["outcome"] == "open"),
        "service": stats([x["visit_duration_s"] for x in services if x["visit_duration_s"] is not None]),
        "desk_unattended_alerts": len([e for e in store.events(types=["alert"])
                                       if e["payload"].get("kind") == "desk_unattended"]),
    }
