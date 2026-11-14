"""`python -m qonaq demo-reel` (S6): 30–45-секундный ролик для продаж.

Состав: заставка → синтетический полный цикл (тот же пайплайн и панель, что в работе) с титрами по событиям
журнала («гости сели», «ждут 3:10 → алерт», «официант подошёл», «стол не убран», «убрали») → кусок реальной
записи CCTV веранды → финальная карточка. Все кадры — из demo.mp4, где лица уже размыты. Поверх всего —
плашка «ДЕМОНСТРАЦИЯ НА ТЕСТОВЫХ ДАННЫХ».
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qonaq.events.store import EventStore
from qonaq.render.sink import VideoSink
from qonaq.render.text import put_text, text_size

log = logging.getLogger("qonaq.reel")
W, H, FPS = 1280, 720, 25
PLAQUE = "ДЕМОНСТРАЦИЯ НА ТЕСТОВЫХ ДАННЫХ"


@dataclass
class Key:
    ts: float  # время сцены, когда показать титр
    text: str


def key_moments(store: EventStore, zone: str = "T1") -> list[Key]:
    """Титры по событиям журнала — в момент, когда событие видно на панели (`detected_ts`), а не в момент,
    к которому логика его потом отнесла (посадка подтверждается через seat_min_s, уход — через leave_min_s)."""
    from qonaq.logic.base import fmt_mmss

    keys: list[Key] = []
    for e in store.events():
        if e["zone_id"] != zone:
            continue
        p = e["payload"]
        seen = float(p.get("detected_ts", e["ts"]))
        if e["type"] == "seated":
            if seen - e["ts"] > 5:
                keys.append(Key(e["ts"], "Гости садятся — проверяем, что это не прохожие"))
            keys.append(Key(seen, "Гости сели — отсчёт ожидания с момента посадки"))
        elif e["type"] == "alert" and p.get("kind") == "no_waiter":
            keys.append(Key(seen, f"Ждут {fmt_mmss(p.get('waited_s', 180))} → алерт администратору в Telegram"))
        elif e["type"] == "served":
            keys.append(Key(seen, f"Официант подошёл: ожидание {fmt_mmss(p.get('response_time_s', 0))}"))
        elif e["type"] == "left":
            # между фактическим уходом и подтверждением панель ещё держит гостей — объясняем почему
            if seen - e["ts"] > 5:
                keys.append(Key(e["ts"], f"Гости вышли — {fmt_mmss(seen - e['ts'])} ждём, не вернутся ли"))
        elif e["type"] == "state_change" and p.get("to") == "NEEDS_CLEANING":
            keys.append(Key(seen, "Гости ушли — стол не убран"))
        elif e["type"] == "cleaned":
            keys.append(Key(seen, f"Стол убрали через {fmt_mmss(p.get('cleanup_time_s', 0))} — снова свободен"))
    return sorted(keys, key=lambda k: k.ts)


def time_map(duration: float, keys: list[Key], target_s: float, slow: float = 5.0, before: float = 3.0,
             after: float = 10.0) -> list[float]:
    """Время сцены для каждого кадра выхода: у ключевых моментов ×slow, между ними — быстрее (сколько нужно)."""
    windows: list[list[float]] = []
    for k in keys:
        a, b = max(0.0, k.ts - before), min(duration, k.ts + after)
        if windows and a <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], b)
        else:
            windows.append([a, b])
    slow_scene = sum(b - a for a, b in windows)
    fast_scene = max(1e-6, duration - slow_scene)
    slow_video = slow_scene / slow
    fast_video = max(2.0, target_s - slow_video)
    fast = max(slow, fast_scene / fast_video)
    out, s = [], 0.0
    dt = 1.0 / FPS
    while s < duration:
        out.append(s)
        in_slow = any(a <= s < b for a, b in windows)
        s += dt * (slow if in_slow else fast)
    return out


def _fit(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    k = min(W / w, H / h)
    r = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), 18, np.uint8)
    y0, x0 = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def _overlay(img: np.ndarray, caption: str | None, sub: str | None = None) -> np.ndarray:
    put_text(img, PLAQUE, (16, 14), size=20, color=(255, 255, 255), bold=True, bg=(40, 40, 200), bg_alpha=0.9,
             pad=8)
    if caption:
        tw, th = text_size(caption, 34, bold=True)
        put_text(img, caption, (W // 2 - tw // 2, H - 96), size=34, color=(255, 255, 255), bold=True,
                 bg=(20, 20, 20), bg_alpha=0.78, pad=12)
    if sub:
        sw, _ = text_size(sub, 20)
        put_text(img, sub, (W // 2 - sw // 2, H - 42), size=20, color=(220, 220, 220), bg=(20, 20, 20),
                 bg_alpha=0.7, pad=6)
    return img


def _card(lines: list[tuple[str, int, bool]]) -> np.ndarray:
    img = np.full((H, W, 3), (34, 28, 22), np.uint8)
    total = sum(int(s * 1.6) for _, s, _ in lines)
    y = (H - total) // 2
    for text, size, bold in lines:
        tw, _ = text_size(text, size, bold)
        put_text(img, text, (W // 2 - tw // 2, y), size=size, color=(255, 255, 255), bold=bold)
        y += int(size * 1.6)
    return _overlay(img, None)


def _frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 2.0)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out, fps


def build_reel(synth_run: Path, real_run: Path | None, out: Path, target_synth_s: float = 26.0,
               real_s: float = 8.0, scene_fps: float = 2.0, real_from: float = 0.0) -> Path:
    """`real_from` — с какой секунды реальной записи брать кусок: в первые ~30 с статусы ещё не установились
    (посадка подтверждается через seat_min_s, «сидели до старта» — через preexisting_grace_s), и ролик показал бы
    людей за «свободными» столами."""
    frames, _ = _frames(synth_run / "demo.mp4")
    if not frames:
        raise FileNotFoundError(f"нет кадров в {synth_run / 'demo.mp4'}")
    store = EventStore(synth_run / "events.sqlite")
    keys = key_moments(store)
    store.close()
    duration = len(frames) / scene_fps
    sink = VideoSink(out, FPS, crf=21)
    for _ in range(int(2.5 * FPS)):
        sink.write(_card([("Qonaq Vision", 64, True), ("Камеры зала считают, сколько гости ждут официанта", 30, False),
                          ("Синтетическая сцена: фигуры вставлены в кадр пустого кафе", 22, False)]))
    for s in time_map(duration, keys, target_synth_s):
        i = min(len(frames) - 1, int(s * scene_fps))
        cur = [k for k in keys if k.ts <= s]
        cap = cur[-1].text if cur else "Стол свободен — ждём гостей"
        _overlay_frame = _overlay(_fit(frames[i]), cap, f"время сцены {int(s) // 60}:{int(s) % 60:02d}")
        sink.write(_overlay_frame)
    if real_run is not None and (real_run / "demo.mp4").exists():
        rf, rfps = _frames(real_run / "demo.mp4")
        n_out = int(real_s * FPS)
        for j in range(n_out):
            i = min(len(rf) - 1, int((real_from + j / FPS) * rfps))
            sink.write(_overlay(_fit(rf[i]), "Реальная запись CCTV веранды кафе", "лица размыты · столы и статусы "
                                                                                    "— автоматически"))
    for _ in range(int(3 * FPS)):
        sink.write(_card([("Видео не уходит из заведения", 34, True), ("Лица не распознаются и размыты", 34, True),
                          ("Бесплатный пилот — 4 недели на ваших камерах", 26, False)]))
    sink.close()
    log.info("демо-ролик: %s (%d кадров, %.1f с)", out, sink.frames, sink.frames / FPS)
    return out
