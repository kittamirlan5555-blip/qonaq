"""Ежедневная сводка владельцу системы (W7): статус, посадки, алерты, 5 кадров с блюром и подписями состояний.

Уходит в отдельный чат (`TELEGRAM_OWNER_CHAT_ID`), не администратору зала. Без токена — печатается/пишется в файл.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

STATE_RU = {"FREE": "свободен", "SEATED_WAITING": "ждут", "SERVED": "обслуживают", "NEEDS_CLEANING": "не убран"}
STATUS_RU = {"ok": "всё в порядке", "warn": "есть предупреждения", "error": "есть ошибки"}


def caption_for(ts_label: str, zone_states: dict[str, dict[str, Any]], max_zones: int = 8) -> str:
    parts = []
    for zid, st in list(zone_states.items())[:max_zones]:
        s = STATE_RU.get(st.get("state", ""), st.get("state", "?"))
        if st.get("state") == "SEATED_WAITING" and st.get("wait_s") is not None:
            s += f" {st['wait_s'] / 60:.0f} мин"
        parts.append(f"{zid}: {s}")
    return f"{ts_label} · " + ", ".join(parts)


class FrameSampler:
    """Откладывает раз в `every_s` кадр с блюром и подписью — для выборочной проверки в сводке."""

    def __init__(self, every_s: float = 600.0, keep: int = 48):
        self.every_s = every_s
        self.samples: deque[tuple[float, bytes, str]] = deque(maxlen=keep)
        self._last: dict[str, float] = {}

    def maybe_add(self, r, blurred_bgr: np.ndarray, ts_label: str | None = None) -> bool:
        if r.ts - self._last.get(r.camera_id, -1e18) < self.every_s:
            return False
        self._last[r.camera_id] = r.ts
        img = blurred_bgr
        if img.shape[1] > 960:
            k = 960 / img.shape[1]
            img = cv2.resize(img, (960, int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            label = ts_label or time.strftime("%H:%M")
            self.samples.append((r.ts, buf.tobytes(), f"{r.camera_id} · " + caption_for(label, r.zone_states)))
        return ok

    def pick(self, n: int = 5) -> list[tuple[bytes, str]]:
        s = list(self.samples)
        if len(s) <= n:
            return [(b, c) for _, b, c in s]
        idx = np.linspace(0, len(s) - 1, n).round().astype(int)
        return [(s[i][1], s[i][2]) for i in idx]


def summary_text(site_name: str, status: dict[str, Any], events: list[dict], alerts: list[dict] | None = None,
                 feedback: list[dict] | None = None, perf: dict[str, Any] | None = None) -> str:
    seated = sum(1 for e in events if e["type"] == "seated")
    served = sum(1 for e in events if e["type"] == "served")
    left = sum(1 for e in events if e["type"] in ("left", "walkout_before_service"))
    walk = sum(1 for e in events if e["type"] == "walkout_before_service")
    alerts = alerts or []
    held = sum(1 for a in alerts if not a.get("confident", True))
    lines = [f"Qonaq · сводка за день · {site_name}",
             f"Статус системы: {STATUS_RU.get(status.get('status', 'ok'), status.get('status'))}",
             f"Посадок: {seated}, первых подходов официанта: {served}, ушли: {left} (не дождавшись: {walk})",
             f"Алертов: {len(alerts)}, из них неуверенных (не отправлены администратору): {held}"]
    if feedback:
        lines.append(f"Отметок «неверно» под алертами: {len(feedback)}")
    for i in status.get("issues", [])[:10]:
        lines.append(f"• {i['camera_id']}: {i['message']}")
    for cid, c in (status.get("cameras") or {}).items():
        if c.get("outages"):
            lines.append(f"• {cid}: обрывов связи {c['outages']}, без картинки всего {c['down_s_total'] / 60:.0f} мин")
    if perf:
        cams = perf.get("cameras") or {}
        fps = ", ".join(f"{k} {v.get('fps', 0):.1f}" for k, v in cams.items())
        lines.append(f"Нагрузка: CPU {perf.get('cpu_pct', 0):.0f}% (100% = одно ядро), память "
                     f"{perf.get('rss_mb') or 0:.0f} МБ; кадр/с по камерам: {fps}")
    lines.append("Ниже — кадры (лица размыты) для выборочной проверки: совпадают ли подписи с тем, что видно?")
    return "\n".join(lines)


def build_daily(run_dir: Path, site_name: str, status: dict[str, Any], frames: list[tuple[bytes, str]],
                perf: dict[str, Any] | None = None) -> dict:
    from qonaq.events.store import EventStore

    store = EventStore(run_dir / "events.sqlite")
    try:
        events = store.events()
        alerts = store.alerts_sent()
        fb = store.feedback()
    finally:
        store.close()
    text = summary_text(site_name, status, events, alerts, fb, perf)
    out = run_dir / "daily"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    (out / f"summary_{stamp}.txt").write_text(text, encoding="utf-8")
    for i, (b, _cap) in enumerate(frames):
        (out / f"frame_{stamp}_{i}.jpg").write_bytes(b)
    (out / f"captions_{stamp}.json").write_text(json.dumps([c for _, c in frames], ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    return {"text": text, "frames": frames, "dir": str(out)}


def send_daily(sender, daily: dict) -> None:
    sender.send(daily["text"])
    if daily["frames"] and hasattr(sender, "send_photos"):
        sender.send_photos(daily["frames"])


def due(now_local: time.struct_time, daily_at: str, last_day: str | None) -> bool:
    """Пора ли слать сводку: местное время прошло daily_at, а сегодня ещё не отправляли."""
    if not daily_at:
        return False
    hh, mm = (int(x) for x in daily_at.split(":"))
    today = time.strftime("%Y-%m-%d", now_local)
    return (now_local.tm_hour, now_local.tm_min) >= (hh, mm) and last_day != today
