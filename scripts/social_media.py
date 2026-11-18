"""Материалы для соцсетей → runs/social/: карусель 1080×1350 и ролик 1080×1920 / 1920×1080.

    python scripts/social_media.py            # всё
    python scripts/social_media.py --only slides
    python scripts/social_media.py --only video

ЧЕСТНОСТЬ. Статусы столов берутся ТОЛЬКО из журнала событий прогона D-046 (`runs/<зал>/d046/events.sqlite`, время
события, а не момент подтверждения) и рисуются ТОЛЬКО в интервалах `VERIFIED`, где они совпадают с ручной разметкой
(`tests/fixtures/gt/*.yaml`). Вне интервала зона стола не рисуется вовсе: ложный статус в кадр не попадает, а
дорисованного события, которого система не видела, нет. Уведомление — реальный алерт прогона (Beach Bar, стол 4,
no_waiter в 15:34), текст — по шаблону бота (`qonaq/strings.py`, tg.no_waiter).

ПРИВАТНОСТЬ. Лица — сильный блюр по ключевым точкам головы (RTMPose) и по верху рамки человека (YOLOX), с запасом и
по двум соседним кадрам; дальний план (бар, толпа, вывески) — расфокус; вывески, логотип трансляции и название бара —
замазаны прямоугольниками `signs`. Видео и картинки — в runs/ (не коммитятся).
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qonaq.events.store import EventStore  # noqa: E402
from qonaq.strings import human_dur, table_label  # noqa: E402
from qonaq.strings import t as tr  # noqa: E402

OUT = ROOT / "runs" / "social"
FPS = 30

# ------------------------------------------------------------------ стиль
GREEN, YELLOW, RED, BLUE, ORANGE = (34, 197, 94), (250, 204, 21), (239, 68, 68), (59, 130, 246), (249, 115, 22)
INK = (17, 20, 26)
BG0, BG1 = (10, 13, 18), (22, 27, 36)
RED_AFTER_S = 180.0  # «ждут дольше 3 мин» — порог алерта no_waiter (SPEC §5)

_FONT_FILES = {
    "r": ["segoeui.ttf", "DejaVuSans.ttf"],
    "sb": ["seguisb.ttf", "DejaVuSans-Bold.ttf"],
    "b": ["segoeuib.ttf", "DejaVuSans-Bold.ttf"],
    "k": ["seguibl.ttf", "DejaVuSans-Bold.ttf"],
    "emoji": ["seguiemj.ttf", "NotoColorEmoji.ttf"],
}
_FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/noto")]
_fcache: dict = {}


def font(size: float, w: str = "sb") -> ImageFont.FreeTypeFont:
    key = (int(size), w)
    if key not in _fcache:
        for name in _FONT_FILES[w]:
            for d in _FONT_DIRS:
                if (d / name).exists():
                    _fcache[key] = ImageFont.truetype(str(d / name), int(size))
                    break
            if key in _fcache:
                break
        else:
            _fcache[key] = ImageFont.load_default(int(size))
    return _fcache[key]


def mmss(s: float) -> str:
    s = max(0, int(s + 0.5))
    return f"{s // 60:02d}:{s % 60:02d}"


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ залы
@dataclass
class Site:
    key: str
    video: str
    file_offset: float  # время файла = время разметки − file_offset
    run: str
    ev_offset: float    # время разметки = ts события + ev_offset
    zones_path: str
    site_path: str
    label_at: dict[str, tuple[float, float]]
    verified: dict[str, list[tuple[float, float]]]
    signs: list[tuple[int, int, int, int]]
    defocus: tuple[int, int]  # выше y0 — расфокус полностью, до y1 — плавно
    stickers: list[tuple[int, int, int, int]] = field(default_factory=list)  # стены в наклейках — мягче, чем вывески
    name: str = ""
    zones: dict[str, list] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def load(self) -> Site:
        z = yaml.safe_load((ROOT / self.zones_path).read_text(encoding="utf-8"))
        self.zones = {d["id"]: d["polygon"] for d in z["zones"]}
        st = EventStore(ROOT / self.run / "events.sqlite")
        try:
            self.events = [dict(e, t=float(e["ts"]) + self.ev_offset) for e in st.events()]
        finally:
            st.close()
        return self


HOGS = Site(
    "hogs", "data/videos/live/hogs_outside_from900.mp4", 900.0, "runs/hogs/d046", 900.0,
    "configs/zones/hogs_outside_seats.yaml", "configs/sites/hogs_outside.yaml",
    label_at={"T01": (1000, 505), "T02": (1450, 870), "T03": (620, 950), "T04": (1770, 640), "T05": (1770, 392)},
    # Секунды файла-источника (как в tests/fixtures/gt/hogs_outside_1116.yaml). Статус системы = разметке:
    verified={
        "T01": [(1080, 1236),     # занят компанией до ухода 20:39 (система «ушли» 21:26 — после 20:36 не рисуем)
                (1577.4, 1760)],  # сели 26:17 (разметка 26:18), подошли 27:34 (27:31); дальше — обслуживают
        "T02": [(1080, 1150)],    # свободен до 19:11; потом система не видит подход (сторона за краем кадра) — не рисуем
        "T03": [(1080, 2880)],    # две гостьи весь отрезок
        "T04": [(1080, 1124),     # заняты до 18:47
                (1127, 1152),     # ушли 18:47 (система 18:45), стаканы забрали только в 19:13 — не убран
                (1217, 2590)],    # протёрли 20:16 → свободен до посадки 43:17 (система «убрано» 19:22 — 19:22–20:16 не рисуем)
        "T05": [(1080, 2880)],    # пуст весь отрезок
    },
    signs=[(1530, 0, 1920, 255), (1150, 0, 1560, 110), (1115, 200, 1240, 285), (1495, 325, 1585, 495),
           (680, 0, 1160, 90)],
    defocus=(300, 390), name="двор бара",
    # стены и колонны в наклейках (чужие логотипы): правая стена, колонна у стола 4, низ стойки слева
    stickers=[(1580, 160, 1920, 395), (1880, 395, 1920, 600), (1862, 600, 1920, 1080), (1460, 250, 1595, 790),
              (0, 610, 250, 1080)])

BEACH = Site(
    "beach", "data/videos/live/beach_bar_st_john_usvi_20261115_2300Z.mp4", 0.0, "runs/beach/d046", 0.0,
    "configs/zones/beach_bar_seats.yaml", "configs/sites/beach_bar.yaml",
    label_at={"B1": (620, 630), "B4": (1760, 565)},
    verified={  # tests/fixtures/gt/beach_bar_holdout.yaml
        "B1": [(300, 1200)],                # пара за столом весь отрезок
        "B4": [(300, 486),                  # компания до ухода 8:08
               (489, 537),                  # ушли 8:08 (система 8:06), убирать начали 8:58 — не убран
               (549, 636),                  # убрали к 9:08 (система 9:11) → свободен
               (762, 1030)],                # сели 12:39 (система 12:34); до 20:00 к ним никто не подходил
    },
    signs=[(0, 0, 1920, 62), (15, 720, 370, 1070), (0, 140, 250, 760), (520, 100, 730, 170), (350, 110, 470, 190),
           (255, 262, 345, 330)],
    defocus=(250, 330), name="бар у моря")
BEACH_T0 = (18 * 3600 + 59 * 60 + 45)  # часы трансляции: 0:00 файла = 18:59:45 (оверлей «7:15:25 PM» на 15:40)


def zone_state(site: Site, zid: str, t: float) -> dict | None:
    """Статус стола в момент t по журналу событий; None — интервал не сверен с разметкой (не рисуем)."""
    if not any(a <= t <= b for a, b in site.verified.get(zid, [])):
        return None
    state, since, served = "FREE", None, None
    for e in site.events:
        if e["zone_id"] != zid or e["t"] > t:
            continue
        if e["type"] == "state_change":
            state = e["payload"]["to"]
        elif e["type"] == "seated":
            since = e["t"]
        elif e["type"] == "left":
            since = e["t"]
        elif e["type"] == "served":
            served = (e["t"], float(e["payload"].get("response_time_s") or 0))
    out = {"state": state}
    if state == "SEATED_WAITING":
        out["timer"] = t - since
    elif state == "NEEDS_CLEANING":
        out["timer"] = t - since
    elif state == "SERVED" and served and t - served[0] < 90:
        out["sub"] = f"официант подошёл через {mmss(served[1])}"
        out["changed"] = served[0]
    for e in site.events:  # момент смены статуса — для «пульса» подписи
        if e["zone_id"] == zid and e["type"] == "state_change" and 0 <= t - e["t"] < 0.8:
            out["pulse"] = (t - e["t"]) / 0.8
    return out


def style_of(st: dict) -> tuple[tuple[int, int, int], str]:
    s = st["state"]
    if s == "FREE":
        return GREEN, "свободен"
    if s == "SEATED_WAITING":
        return (RED if st["timer"] >= RED_AFTER_S else YELLOW), f"ждут {mmss(st['timer'])}"
    if s == "SERVED":
        return BLUE, "обслуживают"
    return ORANGE, f"не убран {mmss(st['timer'])}"


def hall(site: Site, t: float) -> dict:
    st = {z: zone_state(site, z, t) for z in site.zones}
    vis = {z: s for z, s in st.items() if s is not None}
    return {"n": len(vis),
            "occupied": sum(1 for s in vis.values() if s["state"] in ("SEATED_WAITING", "SERVED")),
            "waiting": sum(1 for s in vis.values() if s["state"] == "SEATED_WAITING"),
            "dirty": sum(1 for s in vis.values() if s["state"] == "NEEDS_CLEANING"),
            "free": sum(1 for s in vis.values() if s["state"] == "FREE"),
            "zones": st}


# ------------------------------------------------------------------ приватность
class Privacy:
    def __init__(self):
        from qonaq.config import load_site
        from qonaq.detect.base import create_detector
        from qonaq.pipeline import get_pose_estimator

        cfg = load_site(ROOT / HOGS.site_path).detector
        self.det = create_detector(cfg)
        self.pe = get_pose_estimator(cfg.device)
        self.hist: dict[str, list] = {}
        self.cache: dict = {}

    def heads(self, frame: np.ndarray, boxes) -> list[tuple[float, float, float]]:
        kps = self.pe.estimate(frame, boxes) if boxes else []
        out = []
        for b, k in zip(boxes, kps, strict=False):
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            face = [k[i, :2] for i in range(5) if k[i, 2] >= 0.3]
            r_box = min(max(0.2 * w, 0.08 * h, 9), 0.32 * w + 6)
            if len(face) >= 2:
                c = np.mean(face, axis=0)
                spread = max(np.ptp([p[0] for p in face]), np.ptp([p[1] for p in face]))
                sh = [k[i, :2] for i in (5, 6) if k[i, 2] >= 0.3]
                shw = abs(sh[0][0] - sh[1][0]) if len(sh) == 2 else 0
                r = min(max(spread * 0.95, shw * 0.22, r_box * 0.7, 12), 58)
                if y1 - r <= c[1] <= y1 + 0.65 * h:
                    out.append((float(c[0]), float(c[1]), float(r * 1.35)))
                    continue
            hh = min(0.40 * h, 0.9 * w)
            out.append(((x1 + x2) / 2, y1 + hh * 0.42, min(max(r_box, hh * 0.42), 62) * 1.35))
        return out

    def apply(self, site: Site, frame: np.ndarray, key) -> np.ndarray:
        if key in self.cache:
            return self.cache[key]
        H, W = frame.shape[:2]
        f32 = frame.astype(np.float32)

        def blurred(k: int, sigma: float) -> np.ndarray:
            sm = cv2.resize(frame, (W // k, H // k), interpolation=cv2.INTER_AREA)
            return cv2.resize(cv2.GaussianBlur(sm, (0, 0), sigma), (W, H), interpolation=cv2.INTER_LINEAR)

        # 1) расфокус дальнего плана (бар, толпа, вывески над ним) — плавный переход по высоте
        y0, y1 = site.defocus
        m_bg = np.clip((y1 - np.arange(H, dtype=np.float32)) / max(1, y1 - y0), 0, 1)[:, None].repeat(W, 1)
        # 2) люди у самой камеры (низ кадра, крупно) — целиком: принты и логотипы на одежде читаются
        boxes = [d.xyxy for d in self.det.detect([frame])[0] if d.label == "person"]
        m_fg = np.zeros((H, W), np.float32)
        for x1, yy1, x2, yy2 in boxes:
            if yy2 > H - 60 and yy2 - yy1 > 300:
                cv2.rectangle(m_fg, (int(x1 + 6), int(yy1 + 6)), (int(x2 - 6), int(yy2)), 1.0, -1)
        for x1, yy1, x2, yy2 in site.stickers:
            cv2.rectangle(m_fg, (x1, yy1), (x2, yy2), 1.0, -1)
        m_fg = cv2.GaussianBlur(m_fg, (0, 0), 14)
        # 3) вывески, логотип трансляции, стены в наклейках — сильно, с мягким краем
        m_sg = np.zeros((H, W), np.float32)
        for x1, yy1, x2, yy2 in site.signs:
            cv2.rectangle(m_sg, (x1, yy1), (x2, yy2), 1.0, -1)
        m_sg = cv2.GaussianBlur(m_sg, (0, 0), 16)
        mid = blurred(6, 2.2).astype(np.float32)
        strong = blurred(20, 2.5).astype(np.float32)
        m1 = np.maximum(m_bg, m_fg)[..., None]
        out = f32 * (1 - m1) + mid * m1
        m2 = np.clip(m_sg * 1.25, 0, 1)[..., None]
        img = (out * (1 - m2) + strong * m2).clip(0, 255).astype(np.uint8)
        # 4) лица — сильно, по голове (поза) или по верху рамки; плюс головы с прошлого кадра
        hs = self.heads(frame, boxes)
        prev = self.hist.get(site.key, [])
        for cx, cy, r in hs + prev:
            blur_head(img, cx, cy, r)
        self.hist[site.key] = hs
        if len(self.cache) > 64:
            self.cache.clear()
        self.cache[key] = img
        return img


def blur_head(img: np.ndarray, cx: float, cy: float, r: float) -> None:
    H, W = img.shape[:2]
    R = int(r * 1.5) + 2
    x1, y1, x2, y2 = max(0, int(cx - R)), max(0, int(cy - R)), min(W, int(cx + R)), min(H, int(cy + R))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = img[y1:y2, x1:x2].astype(np.float32)
    h, w = roi.shape[:2]
    k = max(2.0, 2 * r / 4.0)  # голова ≈ 4 пикселя после уменьшения — без деталей лица
    small = cv2.resize(roi, (max(1, int(w / k)), max(1, int(h / k))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 0.9)
    big = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), max(1.0, r / 6))
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(cx - x1), int(cy - y1)), (int(r), int(r * 1.12)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, r * 0.18))[..., None]
    img[y1:y2, x1:x2] = (roi * (1 - m) + big * m).clip(0, 255).astype(np.uint8)


class Reader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(str(ROOT / path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.pos, self.frame = -10**9, None

    def at(self, t: float) -> tuple[np.ndarray, int]:
        idx = int(round(t * self.fps))
        if self.frame is None or idx < self.pos or idx - self.pos > 120:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, self.frame = self.cap.read()
            self.pos = idx
        while self.pos < idx:
            self.cap.grab()
            self.pos += 1
            if self.pos == idx:
                ok, f = self.cap.retrieve()
                if ok:
                    self.frame = f
        return self.frame, idx


# ------------------------------------------------------------------ рисование
def rounded_mask(size: tuple[int, int], r: int) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), r, fill=255)
    return m


def gradient(size: tuple[int, int], c0=BG0, c1=BG1) -> Image.Image:
    w, h = size
    a = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    g = (np.array(c0, np.float32) * (1 - a) + np.array(c1, np.float32) * a)
    return Image.fromarray(np.repeat(g, w, axis=1).astype(np.uint8)).convert("RGBA")


def shadowed(canvas: Image.Image, box, r: int, fill, blur: int = 14, offset: int = 6, shadow_a: int = 110) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    pad = blur * 3
    sh = Image.new("RGBA", (x2 - x1 + 2 * pad, y2 - y1 + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((pad, pad, pad + x2 - x1, pad + y2 - y1), r, fill=(0, 0, 0, shadow_a))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    comp(canvas, sh, x1 - pad, y1 - pad + offset)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle((x1, y1, x2, y2), r, fill=fill)
    canvas.alpha_composite(layer)


def comp(canvas: Image.Image, im: Image.Image, x: int, y: int) -> None:
    """alpha_composite с отрицательным смещением (Pillow требует dest >= 0)."""
    cx, cy = max(0, -x), max(0, -y)
    if cx >= im.width or cy >= im.height:
        return
    canvas.alpha_composite(im.crop((cx, cy, im.width, im.height)), (x + cx, y + cy))


def tables_word(n: int) -> str:
    from qonaq.strings import plural

    return plural(n, "стол", "стола", "столов")


def glass(canvas: Image.Image, box, r: int = 28, tint=(12, 15, 20, 165), border=(255, 255, 255, 38)) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    reg = canvas.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(18))
    reg.alpha_composite(Image.new("RGBA", reg.size, tint))
    canvas.paste(reg, (x1, y1), rounded_mask(reg.size, r))
    lay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(lay).rounded_rectangle((x1, y1, x2 - 1, y2 - 1), r, outline=border, width=2)
    canvas.alpha_composite(lay)


def chaikin(pts: list[tuple[float, float]], it: int = 2) -> list[tuple[float, float]]:
    for _ in range(it):
        out = []
        for i in range(len(pts)):
            p, q = np.array(pts[i]), np.array(pts[(i + 1) % len(pts)])
            out += [tuple(0.75 * p + 0.25 * q), tuple(0.25 * p + 0.75 * q)]
        pts = out
    return pts


@dataclass
class View:
    crop: tuple[float, float, float, float]  # в кадре 1920×1080
    dst: tuple[int, int, int, int]           # x, y, w, h на холсте

    @property
    def s(self) -> float:
        return self.dst[2] / (self.crop[2] - self.crop[0])

    def pt(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.crop[0]) * self.s + self.dst[0], (y - self.crop[1]) * self.s + self.dst[1])


def fit_crop(cx: float, cy: float, w: float, aspect: float, W=1920, H=1080) -> tuple[float, float, float, float]:
    h = w / aspect
    if h > H:
        h, w = H, H * aspect
    x1 = min(max(0, cx - w / 2), W - w)
    y1 = min(max(0, cy - h / 2), H - h)
    return (x1, y1, x1 + w, y1 + h)


def paste_frame(canvas: Image.Image, img: np.ndarray, v: View, radius: int = 0) -> None:
    x1, y1, x2, y2 = v.crop
    x, y, w, h = v.dst
    roi = img[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]
    roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_CUBIC if w > roi.shape[1] else cv2.INTER_AREA)
    pim = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).convert("RGBA")
    canvas.paste(pim, (x, y), rounded_mask((w, h), radius) if radius else None)


def draw_zones(canvas: Image.Image, site: Site, t: float, v: View, label_px: int, clip_r: int = 0,
               only: set[str] | None = None, area=None, keepout=()) -> None:
    """area — где можно рисовать (по умолчанию — окно кадра); keepout — плашки, под которые подписи не заходят."""
    x, y, w, h = v.dst
    if area is not None:
        x, y, w, h = area[0], area[1], area[2] - area[0], area[3] - area[1]
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    hs = hall(site, t)["zones"]
    for zid, st in hs.items():
        if st is None or (only and zid not in only):
            continue
        col, _ = style_of(st)
        pts = chaikin([v.pt(*p) for p in site.zones[zid]])
        d.polygon(pts, fill=(*col, 40))
        d.line(pts + [pts[0]], fill=(*col, 225), width=max(3, label_px // 11), joint="curve")
    clip = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(clip).rounded_rectangle((x, y, x + w - 1, y + h - 1), clip_r, fill=255)
    layer.putalpha(Image.fromarray(np.minimum(np.array(layer.getchannel("A")), np.array(clip))))
    canvas.alpha_composite(layer)
    for zid, st in hs.items():
        if st is None or (only and zid not in only):
            continue
        lx, ly = v.pt(*site.label_at[zid])
        if not (x - 40 <= lx <= x + w + 40 and y - 40 <= ly <= y + h + 40):
            continue
        label(canvas, (lx, ly), zid, st, label_px, (x + 16, y + 16, x + w - 16, y + h - 16), keepout)


def label(canvas: Image.Image, at, zid: str, st: dict, px: int, bounds, keepout=()) -> None:
    col, word = style_of(st)
    pulse = st.get("pulse")
    sc = 1.0 + (0.16 * math.sin(math.pi * pulse) if pulse is not None else 0.0)
    f = font(px * sc, "b")
    text = f"{table_label(zid)} · {word}"
    tw = f.getlength(text)
    ph, pw = int(px * sc * 1.62), int(tw + px * sc * 1.25)
    cx, cy = at
    x1 = min(max(bounds[0], cx - pw / 2), bounds[2] - pw)
    y2 = min(max(bounds[1] + ph, cy - px * 0.35), bounds[3])
    y1 = y2 - ph
    for kx1, ky1, kx2, ky2 in keepout:
        if x1 < kx2 and x1 + pw > kx1 and y1 < ky2 and y2 > ky1:
            x1 = kx2 + 14
    fg = INK if col == YELLOW else (255, 255, 255)
    if col == RED:  # мягкое свечение у «ждут дольше 3 мин»
        glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(glow).rounded_rectangle((x1 - 10, y1 - 10, x1 + pw + 10, y2 + 10), ph, fill=(*RED, 120))
        canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(14)))
    shadowed(canvas, (x1, y1, x1 + pw, y2), ph // 2, (*col, 242), blur=10, offset=4, shadow_a=120)
    d = ImageDraw.Draw(canvas)
    d.text((x1 + pw / 2, y1 + ph / 2), text, font=f, fill=fg, anchor="mm")
    tip = [(cx - px * 0.32, y2 - 1), (cx + px * 0.32, y2 - 1), (cx, y2 + px * 0.34)]
    if x1 + ph / 2 < cx < x1 + pw - ph / 2:
        d.polygon(tip, fill=(*col, 255))
    if st.get("sub"):
        fs = font(px * 0.72, "sb")
        sw = fs.getlength(st["sub"]) + px * 1.0
        sx = min(max(bounds[0], cx - sw / 2), bounds[2] - sw)
        sy1 = y1 - px * 1.35
        shadowed(canvas, (sx, sy1, sx + sw, sy1 + px * 1.12), int(px * 0.56), (255, 255, 255, 240), blur=8, offset=3)
        d.text((sx + sw / 2, sy1 + px * 0.56), st["sub"], font=fs, fill=INK, anchor="mm")


def panel_row(canvas: Image.Image, site: Site, t: float, box, title_px: int = 30) -> None:
    """Плашка «Зал сейчас» в строку: четыре числа."""
    h = hall(site, t)
    glass(canvas, box, r=30)
    x1, y1, x2, y2 = box
    d = ImageDraw.Draw(canvas)
    d.text((x1 + 30, y1 + 26), "Зал сейчас", font=font(title_px, "b"), fill=(255, 255, 255), anchor="lt")
    d.text((x2 - 30, y1 + 30), tables_word(h["n"]),
           font=font(title_px * 0.8, "r"), fill=(200, 205, 215), anchor="rt")
    items = [("занято", h["occupied"], BLUE), ("ждут", h["waiting"], YELLOW), ("не убрано", h["dirty"], ORANGE),
             ("свободно", h["free"], GREEN)]
    cw = (x2 - x1 - 60) / 4
    for i, (name, n, col) in enumerate(items):
        cx = x1 + 30 + cw * i
        ny = y1 + (y2 - y1) * 0.60
        d.text((cx, ny), str(n), font=font(title_px * 2.1, "k"), fill=(255, 255, 255), anchor="lm")
        nw = font(title_px * 2.1, "k").getlength(str(n))
        r = title_px * 0.27
        d.ellipse((cx + nw + 14, ny - r + 2, cx + nw + 14 + 2 * r, ny + r + 2), fill=col)
        d.text((cx + nw + 22 + 2 * r, ny + 3), name, font=font(title_px * 0.85, "sb"), fill=(215, 220, 228),
               anchor="lm")


def panel_col(canvas: Image.Image, site: Site, t: float, box, px: int = 30) -> None:
    """Боковая плашка «Зал сейчас» (горизонтальный ролик)."""
    h = hall(site, t)
    glass(canvas, box, r=30)
    x1, y1, x2, y2 = box
    d = ImageDraw.Draw(canvas)
    d.text((x1 + 30, y1 + 28), "Зал сейчас", font=font(px * 1.15, "b"), fill=(255, 255, 255), anchor="lt")
    d.text((x1 + 30, y1 + 28 + px * 1.5), tables_word(h["n"]),
           font=font(px * 0.75, "r"), fill=(190, 196, 206), anchor="lt")
    items = [("Занято", h["occupied"], BLUE), ("Ждут официанта", h["waiting"], YELLOW),
             ("Не убрано", h["dirty"], ORANGE), ("Свободно", h["free"], GREEN)]
    for i, (name, n, col) in enumerate(items):
        ry = y1 + 28 + px * 3.2 + i * px * 1.75
        d.ellipse((x1 + 30, ry - 9, x1 + 48, ry + 9), fill=col)
        d.text((x1 + 62, ry), name, font=font(px * 0.9, "sb"), fill=(225, 229, 236), anchor="lm")
        d.text((x2 - 30, ry), str(n), font=font(px * 1.25, "k"), fill=(255, 255, 255), anchor="rm")


def wrap(text: str, f, width: float) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        nxt = (cur + " " + w).strip()
        if f.getlength(nxt) <= width or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


def plane_icon(canvas: Image.Image, c, r: float) -> None:
    """Круглая иконка мессенджера: синий круг и бумажный самолётик."""
    cx, cy = c
    d = ImageDraw.Draw(canvas)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(42, 171, 238))
    s = r / 30
    p = [(-15, -1), (17, -13), (11, 15), (3, 7), (-3, 12), (-2, 4), (-15, -1)]
    d.polygon([(cx + x * s, cy + y * s) for x, y in p], fill=(255, 255, 255))
    d.line([(cx + -2 * s, cy + 4 * s), (cx + 12 * s, cy - 9 * s)], fill=(200, 225, 245), width=max(1, int(2 * s)))


def alert_text_b4() -> tuple[str, str]:
    """Реальный алерт прогона: Beach Bar, B4, no_waiter. Текст — шаблон бота tg.no_waiter (qonaq/strings.py)."""
    ev = next(e for e in BEACH.events if e["type"] == "alert" and e["zone_id"] == "B4")
    seated = max(e["t"] for e in BEACH.events if e["type"] == "seated" and e["zone_id"] == "B4" and e["t"] <= ev["t"])
    clk = lambda s: f"{int((BEACH_T0 + s) // 3600) % 24:02d}:{int((BEACH_T0 + s) % 3600 // 60):02d}"  # noqa: E731
    body = tr("tg.no_waiter", zone=table_label("B4"), dur=human_dur(ev["payload"]["waited_s"]), since=clk(seated))
    return clk(ev["t"]), f"{clk(ev['t'])} · {body}"


def toast(canvas: Image.Image, box_xy, width: int, when: str, text: str, px: int = 30, k: float = 1.0) -> int:
    """Всплывающее уведомление в стиле мессенджера. k — прогресс появления 0..1. Возвращает высоту."""
    f = font(px, "r")
    lines = wrap(text, f, width - px * 4.6 - px * 1.4)
    hgt = int(px * 2.9 + len(lines) * px * 1.32)
    x, y = box_xy
    y = int(y - (1 - ease(k)) * (hgt + 40))
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadowed(layer, (x, y, x + width, y + hgt), int(px * 1.1), (250, 250, 252, 248), blur=18, offset=8, shadow_a=140)
    plane_icon(layer, (x + px * 1.9, y + px * 1.9), px * 1.05)
    d = ImageDraw.Draw(layer)
    tx = x + px * 3.6
    d.text((tx, y + px * 0.75), "Бот зала", font=font(px * 1.0, "b"), fill=INK, anchor="lt")
    d.text((x + width - px * 1.0, y + px * 0.8), when, font=font(px * 0.82, "r"), fill=(130, 135, 145), anchor="rt")
    ly = y + px * 2.05
    for i, ln in enumerate(lines):
        if i == 0:
            d.text((tx, ly - px * 0.08), "⚠️", font=font(px * 0.95, "emoji"), embedded_color=True, anchor="lt")
            d.text((tx + px * 1.35, ly), ln, font=f, fill=(35, 38, 45), anchor="lt")
        else:
            d.text((tx, ly), ln, font=f, fill=(35, 38, 45), anchor="lt")
        ly += px * 1.32
    if k < 1:
        a = np.array(layer.getchannel("A"), np.float32) * ease(k)
        layer.putalpha(Image.fromarray(a.astype(np.uint8)))
    canvas.alpha_composite(layer)
    return hgt


def badge(canvas: Image.Image, xy, text: str, px: int = 34, anchor_right: bool = True) -> None:
    f = font(px, "k")
    w = f.getlength(text) + px * 1.1
    h = px * 1.6
    x, y = xy
    if anchor_right:
        x -= w
    glass(canvas, (x, y, x + w, y + h), r=int(h / 2), tint=(12, 15, 20, 175))
    ImageDraw.Draw(canvas).text((x + w / 2, y + h / 2), text, font=f, fill=(255, 255, 255), anchor="mm")


def footer(canvas: Image.Image, xy, px: int = 24, anchor: str = "ms") -> None:
    lay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(lay).text(xy, "демо на записи открытой трансляции", font=font(px, "r"),
                             fill=(255, 255, 255, 140), anchor=anchor)
    canvas.alpha_composite(lay)


def text_block(canvas: Image.Image, xy, title: str, sub: str | None, tp: int, sp: int, width: float,
               alpha: float = 1.0) -> None:
    lay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    x, y = xy
    for ln in wrap(title, font(tp, "b"), width):
        d.text((x, y), ln, font=font(tp, "b"), fill=(255, 255, 255, int(255 * alpha)), anchor="lt")
        y += tp * 1.18
    if sub:
        y += tp * 0.18
        for ln in wrap(sub, font(sp, "r"), width):
            d.text((x, y), ln, font=font(sp, "r"), fill=(195, 202, 214, int(255 * alpha)), anchor="lt")
            y += sp * 1.3
    canvas.alpha_composite(lay)


# ------------------------------------------------------------------ итог (из журнала, только совпавшее с разметкой)
def summary() -> dict:
    """Итог 30-минутного окна Hog's (18:00–48:00 файла) — из событий прогона, совпавших с разметкой.

    Посадки 3 (T02 19:17, T01 26:17, T04 43:15 — все три есть в разметке); первый подход — только у T01 (76,5 с;
    разметка 73 с): у T02 сторона подхода за краем кадра (система 13 мин — ложно), у T04 — не размечается; ушли —
    T04 18:45 и T01 21:26 (разметка 18:47 и 20:39); пик — 4 из 5 столов заняты с 43:15 (T01–T04, верно по разметке)."""
    ev = [e for e in HOGS.events if 1080 <= e["t"] <= 2880]
    seated = [e for e in ev if e["type"] == "seated" and not e["payload"].get("preexisting")]
    served_t01 = next(e for e in ev if e["type"] == "served" and e["zone_id"] == "T01")
    left = [e for e in ev if e["type"] == "left"]
    assert len(seated) == 3 and len(left) == 2, (len(seated), len(left))
    occ = {}
    for t in np.arange(2600, 2880, 10.0):  # занятость по журналу (без фильтра VERIFIED: она верна для всех 5 столов)
        n = 0
        for z in HOGS.zones:
            s = "FREE"
            for e in HOGS.events:
                if e["zone_id"] == z and e["type"] == "state_change" and e["t"] <= t:
                    s = e["payload"]["to"]
            n += s in ("SEATED_WAITING", "SERVED")
        occ[t] = n
    peak = max(occ.values())
    return {"seated": len(seated), "resp": served_t01["payload"]["response_time_s"], "left": len(left), "peak": peak,
            "tables": len(HOGS.zones)}


def summary_card(canvas: Image.Image, box, px: int = 40, alpha: float = 1.0) -> None:
    s = summary()
    x1, y1, x2, y2 = box
    d = ImageDraw.Draw(canvas)
    d.text((x1, y1), "Итог 30 минут", font=font(px * 1.75, "k"), fill=(255, 255, 255), anchor="lt")
    d.text((x1, y1 + px * 2.25), "двор бара · по журналу событий системы", font=font(px * 0.78, "r"),
           fill=(180, 188, 200), anchor="lt")
    cards = [(str(s["seated"]), "посадки гостей за столы", "система увидела каждую", YELLOW),
             (mmss(s["resp"]).lstrip("0"), "до подхода официанта", "стол 1: от посадки до первого подхода", BLUE),
             (str(s["left"]), "стола освободились", "уход гостей отмечен автоматически", ORANGE),
             (f"{s['peak']}/{s['tables']}", "столов заняты в пике", "максимум за 30 минут", GREEN)]
    gy = y1 + px * 3.7
    gap = px * 0.45
    rh = (y2 - gy - 3 * gap) / 4
    nw = max(font(px * 1.9, "k").getlength(c[0]) for c in cards) + px * 1.6
    for i, (big, line1, line2, col) in enumerate(cards):
        cy = gy + i * (rh + gap)
        shadowed(canvas, (x1, cy, x2, cy + rh), int(px * 0.6), (28, 34, 45, 255), blur=12, offset=5)
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((x1 + px * 0.5, cy + rh * 0.22, x1 + px * 0.68, cy + rh * 0.78), int(px * 0.09), fill=col)
        d.text((x1 + px * 1.15, cy + rh / 2), big, font=font(px * 1.9, "k"), fill=(255, 255, 255), anchor="lm")
        d.text((x1 + px * 0.6 + nw, cy + rh / 2 - px * 0.04), line1, font=font(px * 0.86, "b"), fill=(240, 243, 247),
               anchor="ls")
        d.text((x1 + px * 0.6 + nw, cy + rh / 2 + px * 0.34), line2, font=font(px * 0.62, "r"), fill=(165, 174, 188),
               anchor="lt")


# ------------------------------------------------------------------ карусель
def slides(pv: Privacy) -> list[Path]:
    rd = {"hogs": Reader(HOGS.video), "beach": Reader(BEACH.video)}

    def frame(site: Site, t: float) -> np.ndarray:
        f, idx = rd[site.key].at(t - site.file_offset)
        pv.hist.pop(site.key, None)
        return pv.apply(site, f, (site.key, idx))

    out = []
    W, H = 1080, 1350

    # 1 — весь зал и плашка «Зал сейчас»
    t = 1148.0
    c = gradient((W, H))
    text_block(c, (64, 70), "Весь зал — на одном экране", "статус каждого стола обновляется сам, без кассы и кнопок",
               64, 32, W - 128)
    v = View((300, 180, 1920, 1080), (0, 300, 1080, 600))
    paste_frame(c, frame(HOGS, t), v)
    draw_zones(c, HOGS, t, v, 27)
    panel_row(c, HOGS, t, (40, 945, W - 40, 1185), 32)
    lg = [("свободен", GREEN), ("ждут", YELLOW), ("ждут > 3 мин", RED), ("обслуживают", BLUE), ("не убран", ORANGE)]
    d = ImageDraw.Draw(c)
    x = 52
    for name, col in lg:
        d.ellipse((x, 1222, x + 20, 1242), fill=col)
        d.text((x + 28, 1232), name, font=font(25, "sb"), fill=(210, 215, 224), anchor="lm")
        x += 48 + font(25, "sb").getlength(name)
    footer(c, (W / 2, H - 34))
    out.append(save(c, "01_zal_seychas.png"))

    # 2–4 — крупно: ждут / подошли / не убран
    for name, site, t, crop, title, sub in [
        ("02_zhdut_timer.png", HOGS, 1640.0, (640, 180, 1360, 1080), "Гости сели — пошёл таймер",
         "система видит, сколько стол ждёт официанта"),
        ("03_oficiant_podoshel.png", HOGS, 1660.0, (640, 180, 1360, 1080), "Официант подошёл — статус сменился",
         "система записала, сколько стол ждал"),
        ("04_ne_ubran.png", HOGS, 1139.0, (1200, 180, 1920, 1080), "Гости ушли — стол ждёт уборки",
         "таймер простоя идёт, пока стол не уберут"),
    ]:
        c = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        v = View(crop, (0, 0, W, H))
        paste_frame(c, frame(site, t), v)
        shade = gradient((W, 360), (0, 0, 0), (0, 0, 0))
        a = np.linspace(200, 0, 360).astype(np.uint8)[:, None].repeat(W, 1)
        shade.putalpha(Image.fromarray(a))
        c.alpha_composite(shade, (0, 0))
        bot = Image.new("RGBA", (W, 300), (0, 0, 0, 0))
        bot.putalpha(Image.fromarray(np.linspace(0, 190, 300).astype(np.uint8)[:, None].repeat(W, 1)))
        c.alpha_composite(bot, (0, H - 300))
        draw_zones(c, site, t, v, 38, area=(0, 250, W, H - 276))
        text_block(c, (60, 58), title, sub, 58, 31, W - 120)
        panel_row(c, site, t, (36, H - 262, W - 36, H - 70), 28)
        footer(c, (W / 2, H - 26))
        out.append(save(c, name))

    # 5 — коллаж уведомлений
    t = 936.0
    when, text = alert_text_b4()
    c = gradient((W, H))
    text_block(c, (64, 64), "Администратор узнаёт сразу", "уведомление в Telegram, когда к столу 3 минуты не подходят",
               60, 31, W - 128)
    toast(c, (48, 262), W - 96, "сейчас", text, 31)
    v = View((1060, 470, 1920, 800), (48, 470, W - 96, int((W - 96) * 330 / 860)))
    paste_frame(c, frame(BEACH, t), v, radius=30)
    draw_zones(c, BEACH, t, v, 34, clip_r=30, only={"B4"})
    badge(c, (70, 490), "кадр с камеры", 24, anchor_right=False)
    # чат с ботом
    cy = 470 + v.dst[3] + 34
    chat = (48, cy, W - 48, H - 78)
    shadowed(c, chat, 30, (14, 22, 33, 255), blur=14, offset=6)
    d = ImageDraw.Draw(c)
    plane_icon(c, (chat[0] + 52, chat[1] + 50), 28)
    d.text((chat[0] + 96, chat[1] + 30), "Бот зала", font=font(30, "b"), fill=(255, 255, 255), anchor="lt")
    d.text((chat[0] + 96, chat[1] + 66), "бот", font=font(22, "r"), fill=(120, 140, 160), anchor="lt")
    d.line((chat[0], chat[1] + 104, chat[2], chat[1] + 104), fill=(30, 42, 56), width=2)
    f = font(29, "r")
    lines = wrap("⚠ " + text, f, chat[2] - chat[0] - 140)
    bx1, by1 = chat[0] + 30, chat[1] + 128
    bh = 34 + len(lines) * 39 + 26
    bx2 = bx1 + max(f.getlength(ln) for ln in lines) + 120
    d.rounded_rectangle((bx1, by1, bx2, by1 + bh), 22, fill=(24, 37, 51))
    for i, ln in enumerate(lines):
        if i == 0:
            d.text((bx1 + 24, by1 + 24), "⚠️", font=font(27, "emoji"), embedded_color=True, anchor="lt")
            d.text((bx1 + 62, by1 + 22), ln[2:], font=f, fill=(255, 255, 255), anchor="lt")
        else:
            d.text((bx1 + 24, by1 + 22 + i * 39), ln, font=f, fill=(255, 255, 255), anchor="lt")
    d.text((bx2 - 18, by1 + bh - 12), when, font=font(21, "r"), fill=(110, 135, 160), anchor="rb")
    d.rounded_rectangle((bx1, by1 + bh + 10, bx2, by1 + bh + 66), 18, fill=(36, 52, 70))
    d.text(((bx1 + bx2) / 2, by1 + bh + 38), "Неверно", font=font(27, "sb"), fill=(255, 255, 255), anchor="mm")
    footer(c, (W / 2, H - 30))
    out.append(save(c, "05_telegram.png"))

    # 6 — итог
    c = gradient((W, H))
    summary_card(c, (64, 110, W - 64, H - 150), 44)
    footer(c, (W / 2, H - 44))
    out.append(save(c, "06_itog.png"))
    return out


def save(c: Image.Image, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    c.convert("RGB").save(p, optimize=True)
    print(p)
    return p


# ------------------------------------------------------------------ ролик
@dataclass
class Scene:
    site: Site | None
    t0: float
    t1: float
    speed: float
    vcrop: tuple  # вертикаль 4:5 (под шапкой)
    hcrop: tuple  # горизонталь 16:9
    title: str
    sub: str
    xfade: bool = True
    card_s: float = 0.0

    @property
    def dur(self) -> float:
        return self.card_s or (self.t1 - self.t0) / self.speed


def scenes() -> list[Scene]:
    hv, hh = (640, 180, 1360, 1080), (240, 135, 1920, 1080)
    bv, bh = (1200, 180, 1920, 1080), (300, 169, 1920, 1080)
    return [
        Scene(HOGS, 1573, 1581, 2, hv, hh, "Гости садятся за стол", "система видит посадку и запускает таймер"),
        Scene(HOGS, 1581, 1648, 8, hv, hh, "Таймер ожидания официанта", "сколько стол ждёт — видно в секундах",
              xfade=False),
        Scene(HOGS, 1648, 1662, 2, hv, hh, "Официант подошёл", "статус стола сменился, время подхода записано",
              xfade=False),
        Scene(BEACH, 924, 938, 2, bv, bh, "Другой бар: к столу 4 не подходят 3 минуты",
              "администратору уходит уведомление"),
        Scene(None, 0, 0, 1, (), (), "", "", card_s=3.6),
    ]


def render_video(pv: Privacy) -> list[Path]:
    sc = scenes()
    rd = {"hogs": Reader(HOGS.video), "beach": Reader(BEACH.video)}
    when, text = alert_text_b4()
    alert_t = next(e["t"] for e in BEACH.events if e["type"] == "alert" and e["zone_id"] == "B4")
    pipes = {k: ffmpeg(OUT / f"reel_{w}x{h}.mp4", w, h) for k, (w, h) in {"v": (1080, 1920), "h": (1920, 1080)}.items()}
    XF = 15  # кадров на переход
    held: list[tuple[np.ndarray, np.ndarray]] = []

    def frames_of(s: Scene):
        n = int(round(s.dur * FPS))
        for i in range(n):
            if s.site is None:
                if i == 0:
                    cv_, ch_ = card_frame((1080, 1920)), card_frame((1920, 1080))
                yield cv_, ch_
                continue
            t = s.t0 + i / FPS * s.speed
            f, idx = rd[s.site.key].at(t - s.site.file_offset)
            img = pv.apply(s.site, f, (s.site.key, idx))
            ta = (t - alert_t) if s.site is BEACH else None
            yield (vertical(s, img, t, ta, when, text, i / FPS), horizontal(s, img, t, ta, when, text, i / FPS))

    for si, s in enumerate(sc):
        gen = frames_of(s)
        first = []
        if held and s.xfade:
            for (pv_, ph_), (nv, nh) in zip(held, gen, strict=False):
                first.append(1)
                a = len(first) / (len(held) + 1)
                write(pipes, blend(pv_, nv, a), blend(ph_, nh, a))
        elif held:
            for pv_, ph_ in held:
                write(pipes, pv_, ph_)
        held = []
        buf = []
        for fv, fh in gen:
            buf.append((fv, fh))
            if len(buf) > XF:
                write(pipes, *buf.pop(0))
        if si == len(sc) - 1:
            for fv, fh in buf:
                write(pipes, fv, fh)
        else:
            held = buf
        print(f"сцена {si + 1}/{len(sc)} готова", flush=True)
    paths = []
    for p in pipes.values():
        p.stdin.close()
        p.wait()
        paths.append(Path(p.args[-1]))
    return paths


def ffmpeg(path: Path, w: int, h: int) -> subprocess.Popen:
    OUT.mkdir(parents=True, exist_ok=True)
    exe = "ffmpeg"
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def write(pipes, fv: np.ndarray, fh: np.ndarray) -> None:
    pipes["v"].stdin.write(fv.tobytes())
    pipes["h"].stdin.write(fh.tobytes())


def blend(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    k = ease(k)
    return (a.astype(np.float32) * (1 - k) + b.astype(np.float32) * k).astype(np.uint8)


def card_frame(size: tuple[int, int]) -> np.ndarray:
    W, H = size
    c = gradient(size)
    if W < H:
        summary_card(c, (70, 360, W - 70, 1500), 46)
        footer(c, (W / 2, H - 60), 26)
    else:
        summary_card(c, (300, 150, W - 300, H - 120), 42)
        footer(c, (W / 2, H - 40), 24)
    return np.array(c.convert("RGB"))


def speed_text(s: Scene) -> str | None:
    return None if s.speed == 1 else f"×{s.speed:g}"


def vertical(s: Scene, img, t, ta, when, text, local_t) -> np.ndarray:
    W, H = 1080, 1920
    c = gradient((W, H))
    v = View(s.vcrop, (0, 330, 1080, 1350))
    paste_frame(c, img, v)
    draw_zones(c, s.site, t, v, 40)
    a = ease(local_t / 0.5)
    nl = len(wrap(s.title, font(58, "b"), W - 120))
    hb = nl * 58 * 1.18 + 58 * 0.18 + len(wrap(s.sub, font(31, "r"), W - 120)) * 31 * 1.3
    text_block(c, (60, int((330 - hb) / 2) + 6), s.title, s.sub, 58, 31, W - 120, alpha=a)
    if speed_text(s):
        badge(c, (W - 36, 1586), speed_text(s), 38)
    if ta is not None and ta >= 0:
        toast(c, (36, 360), W - 72, "сейчас", text, 31, k=min(1.0, ta / 0.45 / s.speed * 1.0))
    panel_row(c, s.site, t, (36, 1704, W - 36, 1880), 29)
    footer(c, (W / 2, H - 12), 22)
    return np.array(c.convert("RGB"))


def horizontal(s: Scene, img, t, ta, when, text, local_t) -> np.ndarray:
    W, H = 1920, 1080
    c = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    v = View(s.hcrop, (0, 0, W, H))
    paste_frame(c, img, v)
    cap_w = max(font(46, "b").getlength(s.title), font(28, "r").getlength(s.sub)) + 80
    draw_zones(c, s.site, t, v, 34, keepout=[(36, 36, 36 + cap_w, 176), (36, 200, 436, 574)])
    a = ease(local_t / 0.5)
    cap = font(46, "b")
    tw = max(cap.getlength(s.title), font(28, "r").getlength(s.sub)) + 80
    glass(c, (36, 36, 36 + tw, 176), r=30)
    text_block(c, (76, 58), s.title, s.sub, 46, 28, tw, alpha=a)
    panel_col(c, s.site, t, (36, 200, 436, 574), 30)
    if speed_text(s):
        badge(c, (W - 36, 36), speed_text(s), 38)
    if ta is not None and ta >= 0:
        toast(c, (W - 36 - 760, 120), 760, "сейчас", text, 30, k=min(1.0, ta / 0.45 / s.speed))
    footer(c, (W - 40, H - 22), 22, anchor="rs")
    return np.array(c.convert("RGB"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["slides", "video"])
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    HOGS.load()
    BEACH.load()
    pv = Privacy()
    if a.only != "video":
        slides(pv)
    if a.only != "slides":
        for p in render_video(pv):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
