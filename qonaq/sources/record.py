"""Запись с вебкамеры/потока в файл для постановочного теста (W5).

    python -m qonaq record --source webcam:0 --seconds 600 --out data/videos/staged.mp4

Пишется исходное видео (без блюра): это тестовый вход с согласия участников, он остаётся в data/
(в .gitignore) так же, как скачанные тестовые ролики. Всё, что получается из него (демо, кадры), — с блюром.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2

from qonaq.sources.video import parse_source

log = logging.getLogger("qonaq.record")


def webcam_api() -> int:
    """Бэкенд OpenCV для вебкамеры по ОС: macOS — AVFoundation, Windows — DirectShow (быстрее открывается,
    чем MSMF по умолчанию), Linux — по умолчанию (V4L2)."""
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_webcam(index: int, width: int | None = None, height: int | None = None):
    cap = cv2.VideoCapture(index, webcam_api())
    if not cap.isOpened() and webcam_api() != cv2.CAP_ANY:
        cap = cv2.VideoCapture(index)  # запасной путь — бэкенд по умолчанию
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


MAC_HINT = ("macOS: дайте доступ к камере приложению, из которого запускаете (Terminal / iTerm / VS Code): "
            "Системные настройки → Конфиденциальность и безопасность → Камера, затем перезапустите терминал.")


def record(source: str, seconds: float, out: Path, fps: float = 15.0, width: int | None = 1280,
           height: int | None = 720, warmup_s: float = 1.5, preview: bool = False) -> dict:
    from qonaq.render.sink import VideoSink

    kind, target = parse_source(source)
    if kind == "webcam":
        cap = open_webcam(int(target), width, height)
    else:
        cap = cv2.VideoCapture(str(target))
    if not cap.isOpened():
        hint = MAC_HINT if sys.platform == "darwin" else "проверьте индекс камеры (webcam:1) и что она не занята"
        raise OSError(f"камера {source} не открылась. {hint}")
    t_end_warm = time.monotonic() + warmup_s  # первые кадры камеры часто чёрные/пересвеченные
    while time.monotonic() < t_end_warm:
        cap.read()
    ok, frame = cap.read()
    if not ok or frame is None:
        raise OSError(f"камера {source} не отдаёт кадры. {MAC_HINT if sys.platform == 'darwin' else ''}")
    if float(frame.mean()) < 3:
        log.warning("кадр почти чёрный — возможно, нет разрешения на камеру или закрыта шторка")
    out.parent.mkdir(parents=True, exist_ok=True)
    sink = VideoSink(out, fps, crf=20)
    period = 1.0 / fps
    t0 = time.monotonic()
    nxt = t0
    n = 0
    last_log = t0
    try:
        while time.monotonic() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                log.warning("кадр не прочитан")
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if now < nxt:
                continue
            while nxt <= now:  # ровная частота: при подвисании кадр повторяется, время сцены не плывёт
                sink.write(frame)
                n += 1
                nxt += period
            if preview:
                cv2.imshow("qonaq record (q - stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if now - last_log >= 30:
                log.info("запись: %.0f / %.0f с", now - t0, seconds)
                last_log = now
    except KeyboardInterrupt:
        log.info("остановлено")
    finally:
        sink.close()
        cap.release()
        if preview:
            cv2.destroyAllWindows()
    return {"out": str(out), "frames": n, "seconds": round(n / fps, 1), "size": [frame.shape[1], frame.shape[0]]}
