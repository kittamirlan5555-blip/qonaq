"""Редактор зон (SPEC §3.2). Логика (EditorState) отделена от OpenCV-окна и тестируется без GUI.

Управление в окне:
  ЛКМ — добавить вершину      Enter — закрыть полигон     Backspace — убрать вершину / последнюю зону
  t — сменить тип зоны        s — сохранить YAML          q / Esc — выход
  n — переименовать последнюю зону (имя вводится в консоли)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from qonaq.zones.models import ZONE_TYPES, Zone, ZoneSet, save_zones

KEY_ENTER, KEY_BACKSPACE, KEY_ESC = (13, 10), (8, 127), (27,)


class EditorState:
    def __init__(self, zones: ZoneSet | None = None, frame_size: tuple[int, int] | None = None,
                 save_path: str | Path | None = None, camera_id: str = ""):
        self.zones = zones or ZoneSet(camera_id=camera_id, frame_size=frame_size)
        if frame_size and not self.zones.frame_size:
            self.zones.frame_size = frame_size
        self.points: list[tuple[float, float]] = []
        self.zone_type: str = "table"
        self.save_path = Path(save_path) if save_path else None
        self.dirty = False
        self.message = "ЛКМ — вершина, Enter — закрыть, t — тип, s — сохранить, q — выход"

    # --- действия
    def click(self, x: float, y: float) -> None:
        self.points.append((float(x), float(y)))
        self.message = f"вершин: {len(self.points)} (Enter — закрыть полигон)"

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self.message = f"вершина убрана, осталось {len(self.points)}"
        elif self.zones.zones:
            z = self.zones.zones.pop()
            self.dirty = True
            self.message = f"зона {z.id} удалена"

    def close_polygon(self, name: str = "") -> Zone | None:
        if len(self.points) < 3:
            self.message = "нужно минимум 3 вершины"
            return None
        z = Zone(id=self.zones.next_id(self.zone_type), type=self.zone_type, name=name, polygon=list(self.points))
        self.zones.zones.append(z)
        self.points = []
        self.dirty = True
        self.message = f"добавлена зона {z.id} ({z.type})"
        return z

    def cycle_type(self) -> str:
        i = ZONE_TYPES.index(self.zone_type)
        self.zone_type = ZONE_TYPES[(i + 1) % len(ZONE_TYPES)]
        self.message = f"тип зоны: {self.zone_type}"
        return self.zone_type

    def rename_last(self, name: str) -> None:
        if self.zones.zones:
            self.zones.zones[-1].name = name
            self.dirty = True
            self.message = f"{self.zones.zones[-1].id}: «{name}»"

    def save(self) -> Path | None:
        if self.save_path is None:
            self.message = "путь для сохранения не задан"
            return None
        save_zones(self.zones, self.save_path)
        self.dirty = False
        self.message = f"сохранено: {self.save_path}"
        return self.save_path

    def handle_key(self, key: int) -> str:
        """Возвращает действие: '', 'quit', 'rename' (нужен ввод имени)."""
        if key < 0:
            return ""
        key &= 0xFF
        if key in KEY_ENTER:
            self.close_polygon()
        elif key in KEY_BACKSPACE:
            self.undo()
        elif key in (ord("t"), ord("T")):
            self.cycle_type()
        elif key in (ord("s"), ord("S")):
            self.save()
        elif key in (ord("n"), ord("N")):
            return "rename"
        elif key in KEY_ESC or key in (ord("q"), ord("Q")):
            return "quit"
        return ""


def draw_editor(frame: np.ndarray, state: EditorState) -> np.ndarray:
    from qonaq.render.annotate import draw_zones
    from qonaq.render.text import put_text

    img = frame.copy()
    draw_zones(img, state.zones)
    if state.points:
        pts = np.array(state.points, dtype=np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(int(v) for v in p), 4, (0, 255, 255), -1)
    put_text(img, f"[{state.zone_type}] {state.message}", (10, img.shape[0] - 34), 16, (255, 255, 255),
             bg=(30, 30, 30))
    return img


def run_editor(frame: np.ndarray, state: EditorState, max_width: int = 1280) -> EditorState:
    """Интерактивное окно. Требует GUI-сборку OpenCV (opencv-python, не headless)."""
    scale = min(1.0, max_width / frame.shape[1])
    win = "Qonaq — редактор зон"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    except cv2.error as e:  # pragma: no cover
        raise RuntimeError("OpenCV собран без GUI или нет дисплея — правьте YAML зон вручную "
                           "или запустите редактор на машине с экраном") from e

    def on_mouse(event, x, y, *_):  # pragma: no cover — GUI
        if event == cv2.EVENT_LBUTTONDOWN:
            state.click(x / scale, y / scale)

    cv2.setMouseCallback(win, on_mouse)
    while True:  # pragma: no cover — GUI
        img = draw_editor(frame, state)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.imshow(win, img)
        action = state.handle_key(cv2.waitKey(30))
        if action == "rename":
            state.rename_last(input("Имя зоны: ").strip())
        elif action == "quit":
            if state.dirty and state.save_path is not None:
                state.save()
            break
    cv2.destroyAllWindows()
    return state
