"""Запись видео в H.264 mp4 (через ffmpeg из imageio-ffmpeg или системный), запасной путь — OpenCV mp4v.

Сюда попадают только уже обезличенные кадры: сырое видео проект не сохраняет никогда.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("qonaq.render")


def find_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


class VideoSink:
    def __init__(self, path: str | Path, fps: float, crf: int = 23, use_ffmpeg: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(1.0, float(fps))
        self.crf = crf
        self._ffmpeg = find_ffmpeg() if use_ffmpeg else None
        self._proc: subprocess.Popen | None = None
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self.frames = 0

    def _start(self, w: int, h: int) -> None:
        w, h = w - w % 2, h - h % 2  # H.264 требует чётных размеров
        self._size = (w, h)
        if self._ffmpeg:
            cmd = [
                self._ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", f"{self.fps:.3f}", "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", str(self.crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.path),
            ]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                return
            except OSError as e:
                log.warning("ffmpeg не запустился (%s), пишу через OpenCV", e)
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self._size is None:
            self._start(frame.shape[1], frame.shape[0])
        w, h = self._size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = frame[:h, :w] if frame.shape[1] >= w and frame.shape[0] >= h else cv2.resize(frame, (w, h))
        if self._proc is not None:
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError:
                log.error("ffmpeg завершился раньше времени")
                self._proc = None
                return
        elif self._writer is not None:
            self._writer.write(frame)
        self.frames += 1

    def close(self) -> Path:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait(timeout=120)
            self._proc = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self.path

    def __enter__(self) -> VideoSink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
