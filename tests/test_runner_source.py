"""run-file: видео из командной строки главнее источника камеры в --site (раньше молча игнорировалось)."""
from __future__ import annotations

import json

import cv2
import numpy as np


def test_video_argument_overrides_site_camera_source(tmp_path):
    from qonaq.config import CameraConfig, SiteConfig
    from qonaq.detect.base import FakeDetector
    from qonaq.runner import process_file

    clip = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(20):
        w.write(np.zeros((120, 160, 3), np.uint8))
    w.release()
    site = SiteConfig(process_fps=5.0)
    site.cameras = [CameraConfig(id="cam", source=str(tmp_path / "whole_evening.mp4"))]  # такого файла нет
    out = process_file(clip, site, out=tmp_path / "run", detector=FakeDetector(), write_video=False)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["video"] == "clip.mp4" and run["frames"] > 0
    assert site.cameras[0].source.endswith("whole_evening.mp4")  # конфиг вызывающего не тронут
