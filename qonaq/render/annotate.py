"""Отрисовка: (1) отладочный вид — bbox, id треков, роли; (2) демо «как в рилсе» — цветные полигоны
столов с подписями и боковая панель со статусами и таймерами.

Любой кадр, который уходит на диск или в браузер, сначала проходит блюр голов (privacy.blur_faces).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from qonaq.config import SiteConfig
from qonaq.logic.base import fmt_mmss
from qonaq.pipeline import FrameResult
from qonaq.render.blur import blur_people
from qonaq.render.text import put_text, text_size
from qonaq.strings import guests_word, t
from qonaq.zones.models import Zone, ZoneSet

# BGR
GREEN, YELLOW, RED = (90, 200, 80), (0, 210, 255), (60, 60, 235)
BLUE, ORANGE, GRAY, WHITE, DARK = (235, 150, 50), (0, 140, 255), (150, 150, 150), (255, 255, 255), (28, 28, 30)
ROLE_COLOR = {"guest": (120, 220, 120), "staff": (255, 170, 60), "unknown": (200, 200, 200)}
TYPE_COLOR = {"entrance": (220, 120, 220), "waiting_area": (0, 210, 255), "service_point": (235, 150, 50),
              "ignore": (90, 90, 90), "table": GREEN}


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], k: float) -> tuple[int, int, int]:
    k = min(1.0, max(0.0, k))
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))


def state_color(snap: dict[str, Any], no_waiter_s: float = 180.0) -> tuple[int, int, int]:
    """Зелёный FREE, жёлтый SEATED_WAITING (краснеет к порогу), синий SERVED, оранжевый NEEDS_CLEANING."""
    state = snap.get("state")
    if state == "FREE":
        return GREEN
    if state == "SEATED_WAITING":
        wait = snap.get("wait_s", 0.0)
        return _mix(YELLOW, RED, (wait - 0.4 * no_waiter_s) / (0.6 * no_waiter_s))
    if state == "SERVED":
        return RED if snap.get("alert") else BLUE
    if state == "NEEDS_CLEANING":
        return RED if snap.get("alert") else ORANGE
    return TYPE_COLOR.get(snap.get("type", ""), GRAY)


def zone_label(zone: Zone, snap: dict[str, Any] | None) -> str:
    """`T03 · СВОБОДЕН` / `T02 · 2 гостя · 02:14`."""
    name = zone.id
    if not snap:
        return name
    state = snap.get("state")
    if zone.type == "table":
        if state == "FREE":
            return f"{name} · {t('short.FREE')}"
        if state == "SEATED_WAITING":
            g = guests_word(max(1, snap.get("guests", 1)))
            return f"{name} · {g} · {t('panel.waiting')} {fmt_mmss(snap.get('wait_s', 0))}"
        if state == "SERVED":
            return f"{name} · {guests_word(max(1, snap.get('guests', 1)))} · {fmt_mmss(snap.get('session_s', 0))}"
        if state == "NEEDS_CLEANING":
            return f"{name} · {t('short.NEEDS_CLEANING')} {fmt_mmss(snap.get('dirty_s', 0))}"
    if "count" in snap:
        return f"{zone.title} · {snap['count']}"
    return name


def draw_zones(img: np.ndarray, zones: ZoneSet, states: dict[str, dict[str, Any]] | None = None,
               no_waiter_s: float = 180.0, fill_alpha: float = 0.28, font: int = 16,
               labels: bool = True) -> np.ndarray:
    states = states or {}
    overlay = img.copy()
    colors: dict[str, tuple[int, int, int]] = {}
    for z in zones.zones:
        snap = states.get(z.id)
        color = state_color(snap, no_waiter_s) if snap and z.type == "table" else TYPE_COLOR[z.type]
        colors[z.id] = color
        pts = np.array(z.polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        if z.staff_polygon:
            cv2.polylines(img, [np.array(z.staff_polygon, dtype=np.int32)], True, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, dst=img)
    for z in zones.zones:
        pts = np.array(z.polygon, dtype=np.int32)
        cv2.polylines(img, [pts], True, colors[z.id], 2, cv2.LINE_AA)
    if labels:
        for z in zones.zones:
            x1, y1, x2, _ = z.bbox()
            label = zone_label(z, states.get(z.id))
            w, h = text_size(label, font, bold=True)
            put_text(img, label, (int((x1 + x2) / 2 - w / 2), int(y1 - h - 6)), font, WHITE, bold=True,
                     bg=_mix(colors[z.id], DARK, 0.35), bg_alpha=0.85)
    return img


def draw_people(img: np.ndarray, result: FrameResult, show_ids: bool = True, thickness: int = 2,
                font: int = 14) -> np.ndarray:
    for p in result.people:
        x1, y1, x2, y2 = (int(v) for v in p.xyxy)
        color = ROLE_COLOR.get(p.role, GRAY)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if show_ids:
            label = f"#{p.track_id}" + (f" {t('role.' + p.role)}" if p.role != "unknown" else "")
            put_text(img, label, (x1, y2 + 2), font, DARK, bg=color, bg_alpha=0.9, pad=2)
    return img


def render_tracks(result: FrameResult, zones: ZoneSet | None = None, blur: bool = True,
                  top_fraction: float = 0.38, objects: bool = False) -> np.ndarray:
    """Отладочный вид: bbox людей с id и ролями (+ зоны, если есть)."""
    img = result.frame.copy()
    if blur:
        blur_people(img, result.blur_boxes, top_fraction)
    if zones is not None and zones.zones:
        draw_zones(img, zones, result.zone_states)
    if objects:
        for d in result.objects:
            c = (255, 120, 0) if d.label == "chair" else (0, 0, 255)
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), c, 1)
    draw_people(img, result)
    put_text(img, f"{fmt_clock(result.ts)}  {t('panel.people')}: {len(result.people)}", (10, 10), 16, WHITE,
             bg=DARK)
    return img


def fmt_clock(ts: float) -> str:
    """Время сцены: для файлов — мм:сс от начала, для потоков (epoch) — чч:мм:сс по Алматы (UTC+5)."""
    if ts > 1e8:
        from datetime import datetime, timedelta, timezone

        return datetime.fromtimestamp(ts, timezone(timedelta(hours=5))).strftime("%H:%M:%S")
    return fmt_mmss(ts)


class DemoRenderer:
    """Кадр + боковая панель. Высота результата фиксирована (720 по умолчанию)."""

    def __init__(self, zones: ZoneSet, site: SiteConfig | None = None, height: int = 720,
                 panel_width: int = 380, blur: bool = True, show_people: bool = True):
        self.zones = zones
        self.site = site or SiteConfig()
        self.height = height
        self.panel_width = panel_width
        self.blur = blur
        self.show_people = show_people
        self._scaled: ZoneSet | None = None
        self._scale = 1.0
        self.alert_feed: list[tuple[float, str]] = []

    def _prep(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        scale = self.height / h
        if self._scaled is None or abs(scale - self._scale) > 1e-6:
            self._scale = scale
            src = self.zones.scaled(w, h)
            src = src.model_copy(update={"frame_size": (w, h)})
            self._scaled = src.scaled(int(round(w * scale)), self.height)

    def render(self, result: FrameResult) -> np.ndarray:
        from qonaq.strings import alert_text

        self._prep(result.frame)
        s = self._scale
        raw = result.frame
        if self.blur:
            raw = raw.copy()
            blur_people(raw, result.blur_boxes, self.site.privacy.blur_top_fraction)
        fw = int(round(raw.shape[1] * s))
        frame = cv2.resize(raw, (fw - fw % 2, self.height), interpolation=cv2.INTER_AREA)
        draw_zones(frame, self._scaled, result.zone_states, self.site.alerts.no_waiter_s, font=17)
        if self.show_people:
            for p in result.people:
                x1, y1, x2, y2 = (int(v * s) for v in p.xyxy)
                color = ROLE_COLOR.get(p.role, GRAY)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                if p.role == "staff":
                    put_text(frame, t("role.staff"), (x1, y2 + 2), 13, DARK, bg=color, bg_alpha=0.9, pad=2)
        for e in result.events:
            if e.type == "alert":
                self.alert_feed.append((result.ts, alert_text(
                    e.payload.get("kind", ""), e.zone_id,
                    next((v for k, v in e.payload.items() if k.endswith("_s")), 0.0))))
        panel = self._panel(result)
        return np.hstack([frame, panel])

    def _panel(self, result: FrameResult) -> np.ndarray:
        W, H = self.panel_width, self.height
        panel = np.full((H, W, 3), DARK, dtype=np.uint8)
        summ = result.summary
        y = 18
        put_text(panel, t("panel.title"), (20, y), 26, WHITE, bold=True)
        put_text(panel, fmt_clock(result.ts), (W - 20, y + 6), 18, GRAY, anchor="rt")
        y += 50
        tables = self.zones.by_type("table")
        if tables:
            load = summ.get("load_pct", 0)
            put_text(panel, t("panel.load"), (20, y), 16, GRAY)
            put_text(panel, f"{load}%", (W - 20, y - 4), 22, WHITE, bold=True, anchor="rt")
            y += 30
            cv2.rectangle(panel, (20, y), (W - 20, y + 10), (60, 60, 64), -1)
            bar = _mix(GREEN, RED, (load - 50) / 50)
            cv2.rectangle(panel, (20, y), (20 + int((W - 40) * load / 100), y + 10), bar, -1)
            y += 24
            chips = [(t("panel.occupied"), summ.get("occupied", 0), BLUE),
                     (t("panel.waiting"), summ.get("waiting", 0), YELLOW),
                     (t("panel.dirty"), summ.get("dirty", 0), ORANGE),
                     (t("panel.free"), summ.get("free", 0), GREEN)]
            cw = (W - 40) // 4
            for i, (label, n, color) in enumerate(chips):
                x = 20 + i * cw
                put_text(panel, str(n), (x, y), 24, color, bold=True)
                put_text(panel, label, (x, y + 30), 12, GRAY)
            y += 62
        cv2.line(panel, (20, y), (W - 20, y), (60, 60, 64), 1)
        y += 10
        rows = [z for z in self.zones.zones if z.type != "ignore"]
        feed_h = 110 if self.alert_feed else 0
        avail = H - y - 14 - feed_h
        row_h = int(min(40, max(20, avail / max(1, len(rows)))))
        fs = 16 if row_h >= 30 else 13
        for z in rows:
            snap = result.zone_states.get(z.id, {})
            color = state_color(snap, self.site.alerts.no_waiter_s) if z.type == "table" else TYPE_COLOR[z.type]
            cy = y + row_h // 2
            cv2.circle(panel, (30, cy), 6, color, -1, cv2.LINE_AA)
            put_text(panel, z.id, (46, cy - fs // 2 - 3), fs, WHITE, bold=True)
            status, timer = self._row_text(z, snap)
            put_text(panel, status, (104, cy - fs // 2 - 3), fs, color if z.type == "table" else WHITE)
            put_text(panel, timer, (W - 20, cy - fs // 2 - 3), fs, WHITE, anchor="rt")
            y += row_h
        if self.alert_feed:
            y0 = H - feed_h
            cv2.line(panel, (20, y0), (W - 20, y0), (60, 60, 64), 1)
            for i, (_, text) in enumerate(self.alert_feed[-3:][::-1]):
                put_text(panel, "⚠ " + text, (20, y0 + 10 + i * 30), 13, RED if i == 0 else GRAY)
        return panel

    @staticmethod
    def _row_text(z: Zone, snap: dict[str, Any]) -> tuple[str, str]:
        state = snap.get("state", "")
        if z.type == "table":
            status = t(f"state.{state}") if state else ""
            if state == "SEATED_WAITING":
                return status, fmt_mmss(snap.get("wait_s", 0))
            if state == "SERVED":
                g = snap.get("guests", 0)
                return f"{status} · {g}" if g else status, fmt_mmss(snap.get("session_s", 0))
            if state == "NEEDS_CLEANING":
                return status, fmt_mmss(snap.get("dirty_s", 0))
            return status, fmt_mmss(snap.get("state_s", 0))
        label = z.name or t(f"zone.{z.type}")
        if z.type == "service_point":
            staffed = snap.get("staffed")
            desk = "" if staffed is None else (" · сотрудник есть" if staffed else " · нет сотрудника")
            return f"{label}{desk}", str(snap.get("count", 0))
        return label, str(snap.get("count", 0))
