"""Простои по журналу (B7): когда система не работала и какие камеры не давали кадров — по событиям heartbeat.

live пишет heartbeat раз в `health.stats_every_s` (окно и камеры с кадрами за окно), при штатной остановке —
последний с `final: true`. Отсюда, без health.json текущего процесса (он перезаписывается при перезапуске):
- разрыв между heartbeat'ами длиннее окна → система не работала: перезапуск службы, выключение ПК, питание.
  Если перед разрывом был `final` — остановка штатная, иначе — аварийная (процесс убит, питание, зависание);
- камера отсутствует в `up` несколько окон подряд → камера не давала кадров.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo


@dataclass
class Gap:
    start: float
    end: float
    camera_id: str = ""  # "" — вся система
    clean: bool | None = None  # для системы: штатная остановка перед разрывом (None — неизвестно)

    @property
    def minutes(self) -> float:
        return (self.end - self.start) / 60.0


def downtime(events: list[dict], since: float | None = None, until: float | None = None,
             slack_s: float = 15.0, min_cam_s: float = 50.0) -> list[Gap]:
    """Простои системы и камер по heartbeat'ам из `events` (остальные события игнорируются)."""
    hbs = sorted((e for e in events if e.get("type") == "heartbeat"), key=lambda e: e["ts"])
    if since is not None:
        hbs = [e for e in hbs if e["ts"] >= since]
    if until is not None:
        hbs = [e for e in hbs if e["ts"] <= until]
    if not hbs:
        return []
    cams = sorted({c for e in hbs for c in ((e.get("payload") or {}).get("up") or [])})
    gaps: list[Gap] = []
    prev = None
    open_down: dict[str, float] = {}

    def close_cam(cid: str, end: float) -> None:
        a = open_down.pop(cid, None)
        if a is not None and end - a >= min_cam_s:
            gaps.append(Gap(a, end, cid))

    for e in hbs:
        p = e.get("payload") or {}
        w = float(p.get("window_s", 60.0))
        a = e["ts"] - w
        if prev is not None and a - prev["ts"] > slack_s:
            pp = prev.get("payload") or {}
            gaps.append(Gap(prev["ts"], a, "", clean=bool(pp.get("final"))))
            for cid in list(open_down):  # во время разрыва системы камеру отдельно не считаем
                close_cam(cid, prev["ts"])
        up = set(p.get("up") or [])
        final = bool(p.get("final"))
        for cid in cams:
            if cid in up or final:  # последнее окно при остановке — не «камера пропала»
                if cid in open_down:
                    close_cam(cid, a)
            elif cid not in open_down:
                open_down[cid] = a
        prev = e
    for cid in list(open_down):
        close_cam(cid, prev["ts"])
    return sorted(gaps, key=lambda g: (g.start, g.camera_id))


def _hm(ts: float, tz: tzinfo | None) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%d.%m %H:%M")


def downtime_lines(gaps: list[Gap], tz: tzinfo | None = None, limit: int = 10) -> list[str]:
    out = []
    for g in gaps[:limit]:
        span = f"{_hm(g.start, tz)}–{_hm(g.end, tz)[-5:]} ({g.minutes:.0f} мин)"
        if g.camera_id:
            out.append(f"{g.camera_id}: не было кадров {span}")
        else:
            how = "остановлена штатно" if g.clean else "аварийно: сбой, выключение или питание"
            out.append(f"система не работала {span} — {how}")
    if len(gaps) > limit:
        out.append(f"… и ещё {len(gaps) - limit}")
    return out
