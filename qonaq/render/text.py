"""Текст с кириллицей поверх кадра. cv2.putText кириллицу не умеет, поэтому рисуем через Pillow.

Каждая строка рендерится один раз в маску uint8 и кэшируется — на кадр уходит только альфа-смешивание.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
_BOLD = {
    "segoeui.ttf": "segoeuib.ttf", "arial.ttf": "arialbd.ttf", "DejaVuSans.ttf": "DejaVuSans-Bold.ttf",
    "Arial.ttf": "Arial Bold.ttf",
}


@lru_cache(maxsize=1)
def _font_path() -> str | None:
    for c in _CANDIDATES:
        if Path(c).exists():
            return c
    # matplotlib (зависимость ultralytics) везёт DejaVuSans с кириллицей
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.origin:
        p = Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
        if p.exists():
            return str(p)
    return None


@lru_cache(maxsize=64)
def get_font(size: int, bold: bool = False):
    path = _font_path()
    if path is None:
        return ImageFont.load_default()
    if bold:
        b = Path(path).with_name(_BOLD.get(Path(path).name, Path(path).name))
        if b.exists():
            path = str(b)
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=1024)
def _alpha(text: str, size: int, bold: bool) -> np.ndarray:
    """Маска текста uint8 (h, w). Кэш ограничен и по числу (1024), и по весу: uint8 вместо float32 RGBA —
    в 16 раз меньше. Раньше в кэше лежали float32-патчи на 4096 строк: подписи с таймерами («ждут 03:12»)
    все разные, и живой режим набирал до ~0,5 ГБ (S1, EVAL §14)."""
    font = get_font(size, bold)
    l, t, r, b = font.getbbox(text)
    w, h = max(1, r - l + 2), max(1, int(size * 1.3))
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((1 - l, 0), text, fill=255, font=font)
    return np.asarray(img, dtype=np.uint8)


def text_size(text: str, size: int, bold: bool = False) -> tuple[int, int]:
    a = _alpha(text, size, bold)
    return a.shape[1], a.shape[0]


def put_text(img: np.ndarray, text: str, org: tuple[int, int], size: int = 18,
             color: tuple[int, int, int] = (255, 255, 255), bold: bool = False,
             bg: tuple[int, int, int] | None = None, bg_alpha: float = 0.75, pad: int = 4,
             anchor: str = "lt") -> tuple[int, int, int, int]:
    """Рисует text; org — левый верхний угол (anchor='lt') или центр ('c'). Возвращает занятый bbox."""
    if not text:
        return (org[0], org[1], org[0], org[1])
    mask = _alpha(text, size, bold)
    h, w = mask.shape[:2]
    x, y = int(org[0]), int(org[1])
    if anchor == "c":
        x, y = x - w // 2, y - h // 2
    elif anchor == "rt":
        x = x - w
    H, W = img.shape[:2]
    x = min(max(x, pad), max(pad, W - w - pad))
    y = min(max(y, pad), max(pad, H - h - pad))
    if bg is not None:
        bx1, by1, bx2, by2 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        roi = img[by1:by2, bx1:bx2].astype(np.float32)
        roi = roi * (1 - bg_alpha) + np.array(bg, dtype=np.float32) * bg_alpha
        img[by1:by2, bx1:bx2] = roi.astype(np.uint8)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x or y2 <= y:
        return (x, y, x, y)
    a = mask[: y2 - y, : x2 - x, None].astype(np.float32) / 255.0
    roi = img[y:y2, x:x2].astype(np.float32)
    img[y:y2, x:x2] = (roi * (1 - a) + np.array(color, dtype=np.float32) * a).astype(np.uint8)
    return (x - pad, y - pad, x + w + pad, y + h + pad)
