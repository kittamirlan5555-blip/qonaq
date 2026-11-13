"""Общие типы логики: наблюдения на входе, события на выходе. Логика не знает про видео."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ZoneObservation:
    """Что видно в зоне в момент ts (секунды времени сцены).

    Роли уже определены слоем roles: guest / staff / unknown. id треков нужны только для визитов
    и dwell — занятость считается по присутствию (SPEC §9a п.1).
    """

    ts: float
    zone_id: str
    guest_tracks: tuple[int, ...] = ()
    staff_tracks: tuple[int, ...] = ()
    unknown_tracks: tuple[int, ...] = ()
    # Кто из людей в зоне стоит (по позе/пропорциям bbox). None — признак недоступен.
    # SPEC §9a п.3: без калибровки формы визитом персонала считается только визит СТОЯЩЕГО неизвестного.
    standing_tracks: tuple[int, ...] | None = None
    # Сигнал по эталону пустого стола: True — на столе что-то есть, False — стол как эталон, None — нет данных.
    table_dirty: bool | None = None
    # W3: прочие сигналы состояния стола [(имя, True/False/None)] — clf, dishes. Сливаются в логике.
    table_signals: tuple[tuple[str, bool | None], ...] | None = None
    # W8: средняя уверенность ролей «гость» в зоне на этом кадре (None — гостей нет / неизвестно).
    guest_confidence: float | None = None
    # service_point: сотрудник за стойкой (clinic).
    staff_behind_desk: bool | None = None


@dataclass
class Event:
    ts: float  # момент, к которому относится событие (может быть «задним числом»)
    zone_id: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    camera_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": round(self.ts, 3),
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "type": self.type,
            "payload": self.payload,
        }


class PresenceFilter:
    """Скользящее окно присутствия с гистерезисом (SPEC §9a п.1).

    Включается, когда доля положительных наблюдений за окно >= on_ratio; выключается при доле <= off_ratio.
    `since` — момент, с которого текущее значение фактически верно (задним числом: первое положительное
    наблюдение в окне при включении, последнее положительное — при выключении).
    """

    def __init__(self, window_s: float = 5.0, on_ratio: float = 0.7, off_ratio: float = 0.2):
        self.window_s = window_s
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.state = False
        self.since: float | None = None
        self._obs: deque[tuple[float, bool]] = deque()
        self._first_ts: float | None = None
        self._last_positive: float | None = None

    def update(self, ts: float, present: bool) -> bool:
        """Добавляет наблюдение; возвращает True, если состояние изменилось."""
        if self._first_ts is None:
            self._first_ts = ts
            self.since = ts
        self._obs.append((ts, present))
        while self._obs and self._obs[0][0] < ts - self.window_s:
            self._obs.popleft()
        positives = sum(1 for _, p in self._obs if p)
        ratio = positives / len(self._obs)
        changed = False
        if not self.state:
            span = min(ts - self._first_ts, self.window_s)
            if positives >= 2 and ratio >= self.on_ratio and span >= self.on_ratio * self.window_s - 1e-6:
                self.state = True
                self.since = next(t for t, p in self._obs if p)
                changed = True
        elif ratio <= self.off_ratio:
            self.state = False
            self.since = self._last_positive if self._last_positive is not None else ts
            changed = True
        if present:
            self._last_positive = ts
        return changed

    def duration(self, ts: float) -> float:
        return ts - self.since if self.since is not None else 0.0


def fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
