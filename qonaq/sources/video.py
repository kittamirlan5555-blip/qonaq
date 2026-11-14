"""VideoSource: файл | webcam:N | rtsp://... -> (frame, ts_seconds) с семплингом до process_fps.

Время сцены: для файлов — время видео (по номеру кадра / CAP_PROP_POS_MSEC), для потоков — wall clock.
Для потоков кадры читает фоновый поток и отдаёт только самый свежий (иначе буфер RTSP даёт задержку);
при обрыве — переподключение с экспоненциальной задержкой.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from qonaq.config import expand_env, mask_secrets, unresolved_env

log = logging.getLogger("qonaq.source")

Frame = np.ndarray


@dataclass
class SourceInfo:
    kind: str  # file | webcam | stream
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 and self.frame_count > 0 else 0.0


def parse_source(spec: str) -> tuple[str, str | int]:
    """'webcam:0' -> ('webcam', 0); 'rtsp://..' -> ('stream', url); путь -> ('file', path)."""
    missing = unresolved_env(str(spec))
    if missing:
        log.error("в адресе камеры не подставлены %s: нет в .env (запускайте из папки проекта или проверьте .env)",
                  ", ".join("${" + m + "}" for m in missing))
    spec = expand_env(str(spec)).strip()
    low = spec.lower()
    if low.startswith("webcam"):
        idx = low.split(":", 1)[1] if ":" in low else "0"
        return "webcam", int(idx or 0)
    if spec.isdigit():
        return "webcam", int(spec)
    if "://" in spec:
        return "stream", spec
    return "file", spec


class SteadyClock:
    """Время потоков (B8): эпоха, но идущая по монотонным часам — не прыгает назад при подводке часов ПК.

    Раньше кадр/с ограничивался по time.time(): откат часов (NTP после загрузки мини-ПК без батарейки, ручная
    подводка) на час останавливал обработку камеры на час, а небольшие откаты давали отрицательные интервалы в
    логике (таймеры «ушли»/«не убран» не срабатывали). Теперь: t = эпоха старта + монотонное время.
    Если часы ПК ушли **вперёд** больше чем на jump_s — переякориться (в логике это разрыв, как обрыв камеры).
    Если **назад** — время событий продолжает идти монотонно (опережает часы ПК), в лог — предупреждение;
    после перезапуска службы — снова по часам ПК."""

    def __init__(self, wall=time.time, mono=time.monotonic, jump_s: float = 60.0):
        self._wall, self._mono, self.jump_s = wall, mono, jump_s
        self._lock = threading.Lock()
        self._w0, self._m0 = wall(), mono()
        self._warned_back = 0.0
        self.jumps = 0

    def __call__(self) -> float:
        with self._lock:
            m = self._mono()
            t = self._w0 + (m - self._m0)
            d = self._wall() - t
            if d > self.jump_s:
                log.warning("часы ПК ушли вперёд на %.0f с — дальше время событий по новым часам", d)
                self._w0, self._m0 = t + d, m
                self.jumps += 1
                return t + d
            if d < -self.jump_s and m - self._warned_back > 600:
                self._warned_back = m
                log.warning("часы ПК ушли назад на %.0f с — время событий идёт монотонно (опережает часы ПК) до "
                            "перезапуска службы; проверьте синхронизацию времени (NTP)", -d)
            return t


steady_clock = SteadyClock()


def backoff_delays(base: float = 1.0, factor: float = 2.0, max_delay: float = 15.0) -> Iterator[float]:
    d = base
    while True:
        yield min(d, max_delay)
        d *= factor


def open_stream(url: str, timeout_s: float = 10.0):
    """RTSP/HTTP-поток через FFmpeg-бэкенд OpenCV: TCP-транспорт и таймауты открытия/чтения.

    Без таймаутов OpenCV на мёртвом регистраторе висит в read() ~30 с и дольше. TCP вместо UDP — меньше
    рассыпающихся кадров через коммутаторы/Wi-Fi. Переопределяется переменной OPENCV_FFMPEG_CAPTURE_OPTIONS.
    """
    import os

    if url.lower().startswith("rtsp") and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    ms = int(timeout_s * 1000)
    try:
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms])
    except (cv2.error, TypeError, AttributeError):  # старый OpenCV без параметров открытия
        return cv2.VideoCapture(url)


def probe(path: str | Path) -> SourceInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f"не удалось открыть видео: {path}")
        return SourceInfo(
            kind="file",
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
    finally:
        cap.release()


class VideoSource:
    """Итератор по (frame, ts). `read()` возвращает None, когда источник закончился (только файлы)."""

    def __init__(
        self,
        spec: str,
        process_fps: float = 5.0,
        loop: bool = False,
        realtime: bool | None = None,
        max_seconds: float | None = None,
        opener=None,
        sleep=time.sleep,
        clock=steady_clock,
        timeout_s: float = 10.0,
    ):
        self.spec = spec
        self.kind, self.target = parse_source(spec)
        self.process_fps = process_fps
        self.loop = loop
        # Файл по умолчанию обрабатывается так быстро, как можем; потоки — всегда в реальном времени.
        self.realtime = (self.kind != "file") if realtime is None else realtime
        self.max_seconds = max_seconds
        self._open = opener or self._default_opener
        self._sleep = sleep
        self._clock = clock
        self._cap = None
        self.info = SourceInfo(kind=self.kind)
        self.reconnects = 0
        # file state
        self._frame_idx = -1
        self._next_ts = 0.0
        self._ts_offset = 0.0
        self._last_video_ts = 0.0
        self._wall_start: float | None = None
        # stream state
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: tuple[Frame, float] | None = None
        self._stop = threading.Event()
        self._last_emit = 0.0
        self.frames_received = 0
        self.last_frame_wall: float | None = None
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ open/close
    def _default_opener(self):
        if self.kind == "webcam":
            from qonaq.sources.record import open_webcam  # AVFoundation на macOS, DirectShow на Windows

            cap = open_webcam(int(self.target))
        elif self.kind == "stream":
            cap = open_stream(str(self.target), self.timeout_s)
        else:
            cap = cv2.VideoCapture(str(self.target))
        if self.kind != "file":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def open(self) -> VideoSource:
        if self.kind == "file":
            if not Path(str(self.target)).exists():
                raise FileNotFoundError(f"видео не найдено: {self.target}")
            self._cap = self._open()
            if not self._cap.isOpened():
                raise OSError(f"не удалось открыть видео: {self.target}")
            self._read_info(self._cap)
        else:
            # Подключение — в фоновом потоке: недоступная при старте камера не держит остальные (S1).
            self._thread = threading.Thread(target=self._reader_main, name=f"src-{self.kind}", daemon=True)
            self._thread.start()
        return self

    def _read_info(self, cap) -> None:
        self.info = SourceInfo(
            kind=self.kind,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if self.kind == "file" else 0,
        )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ streams
    def _connect_with_retries(self, first: bool = False) -> bool:
        """Подключение с экспоненциальной задержкой. Возвращает False, если попросили остановиться."""
        for delay in backoff_delays():
            if self._stop.is_set():
                return False
            cap = self._open()
            if cap is not None and cap.isOpened():
                self._cap = cap
                self._read_info(cap)
                if not first:
                    self.reconnects += 1
                    log.info("источник %s: переподключено", mask_secrets(str(self.target)))
                return True
            if cap is not None:
                cap.release()
            log.warning("источник %s недоступен, повтор через %.0f с", mask_secrets(str(self.target)), delay)
            if self._stop.wait(delay) if self._sleep is time.sleep else self._sleep(delay):
                return False
        return False

    def _reader_main(self) -> None:
        if self._connect_with_retries(first=True):
            self._reader_loop()

    @property
    def connected(self) -> bool:
        return self._cap is not None

    def _reader_loop(self) -> None:
        fails = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read() if self._cap is not None else (False, None)
            if not ok or frame is None:
                fails += 1
                if fails >= 10:
                    log.warning("источник %s: поток оборвался", mask_secrets(str(self.target)))
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                    if not self._connect_with_retries():
                        return
                    fails = 0
                else:
                    time.sleep(0.05)
                continue
            fails = 0
            now = self._clock()
            with self._lock:
                self._latest = (frame, now)
                self.frames_received += 1
                self.last_frame_wall = now

    def poll(self) -> tuple[Frame, float] | None:
        """Неблокирующее чтение потока: свежий кадр, если он есть и пора (process_fps), иначе None сразу.

        Для файлов то же, что read(). Нужен многокамерному циклу: пропавшая камера не держит остальные.
        """
        if self.kind == "file":
            return self._read_file()
        now = self._clock()
        if now < self._last_emit + 1.0 / self.process_fps:
            return None
        with self._lock:
            item, self._latest = self._latest, None
        if item is not None:
            self._last_emit = now
        return item

    def _read_stream(self) -> tuple[Frame, float] | None:
        period = 1.0 / self.process_fps
        deadline = self._clock() + 30.0
        while not self._stop.is_set():
            now = self._clock()
            wait = self._last_emit + period - now
            if wait > 0:
                time.sleep(min(wait, 0.05))
                continue
            with self._lock:
                item, self._latest = self._latest, None
            if item is not None:
                self._last_emit = now
                return item
            if now > deadline:
                deadline = now + 30.0
                log.warning("источник %s: нет кадров 30 с", mask_secrets(str(self.target)))
            time.sleep(0.01)
        return None

    # ------------------------------------------------------------------ files
    def _read_file(self) -> tuple[Frame, float] | None:
        fps = self.info.fps if self.info.fps > 0 else 25.0
        while True:
            if not self._cap.grab():
                if self.loop:
                    self._ts_offset = self._last_video_ts + 1.0 / fps
                    self._cap.release()
                    self._cap = self._open()
                    self._frame_idx = -1
                    if not self._cap.isOpened() or not self._cap.grab():
                        return None
                else:
                    return None
            self._frame_idx += 1
            # Время видео: CAP_PROP_POS_MSEC корректно и для переменной частоты кадров; запасной путь — номер кадра.
            pos = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if pos <= 0.0 and self._frame_idx > 0:
                pos = self._frame_idx / fps
            ts = self._ts_offset + max(pos, self._last_video_ts - self._ts_offset if self._frame_idx else 0.0)
            self._last_video_ts = ts
            if self.max_seconds is not None and ts > self.max_seconds:
                return None
            if ts + 1e-9 < self._next_ts:
                continue
            ok, frame = self._cap.retrieve()
            if not ok or frame is None:
                continue
            period = 1.0 / self.process_fps
            self._next_ts = max(self._next_ts + period, ts - period / 2 + period)
            if self.realtime:
                if self._wall_start is None:
                    self._wall_start = self._clock() - ts
                wait = self._wall_start + ts - self._clock()
                if wait > 0:
                    self._sleep(wait)
            return frame, ts

    # ------------------------------------------------------------------ API
    def read(self) -> tuple[Frame, float] | None:
        if self.kind == "file":
            return self._read_file()
        return self._read_stream()

    def __iter__(self) -> Iterator[tuple[Frame, float]]:
        while True:
            item = self.read()
            if item is None:
                return
            yield item
