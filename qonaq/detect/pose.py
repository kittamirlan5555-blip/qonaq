"""Поза по ключевым точкам: RTMPose (OpenMMLab, Apache-2.0) через onnxruntime, «сверху вниз» по рамкам людей.

Зачем (EVAL §26, D-044): отношение сторон рамки не отличает «сидит» от «нагнулся к столу» и от «стоит за столом» —
официант, нагнувшийся протереть стол, становился «сидящим гостем». Здесь — 17 точек COCO на человека и правило
`posture()`: сидит / стоит / нагнулся / неизвестно.

Веса: rtmpose-m body7 256×192 (ONNX из релиза mmpose, ~50 МБ) → `data/weights/` (как YOLOX, в git не идут).
Лицензия кода и весов — Apache-2.0; среди датасетов обучения body7 есть с некоммерческими условиями (AI Challenger) —
перед продажей проверить (DECISIONS D-044). Модель за интерфейсом `PoseEstimator`: замена — один класс.
"""
from __future__ import annotations

import logging
import math
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("qonaq.pose")

POSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")
POSE_FILE = "rtmpose_m_body7_256x192.onnx"
W, H = 192, 256
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
# COCO-17
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16

SIT, STAND, BEND = "sit", "stand", "bend"


def ensure_pose_weights() -> Path:
    from qonaq.detect.onnx_yolox import weights_dir

    dst = weights_dir() / POSE_FILE
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    log.info("скачиваю веса позы %s ...", POSE_URL)
    tmp = dst.with_suffix(".zip")
    urllib.request.urlretrieve(POSE_URL, tmp)  # noqa: S310 — фиксированный https-адрес релиза mmpose
    with zipfile.ZipFile(tmp) as z:
        name = next(n for n in z.namelist() if n.endswith("end2end.onnx"))
        dst.write_bytes(z.read(name))
    tmp.unlink()
    return dst


class PoseEstimator:
    """Ключевые точки людей по рамкам. `estimate(frame, boxes)` → (N, 17, 3): x, y (пиксели кадра), уверенность."""

    name = "rtmpose_m"

    def __init__(self, device: str = "auto", batch: int = 16):
        import onnxruntime as ort

        from qonaq.detect.onnx_yolox import device_from_providers, select_providers

        self.session = ort.InferenceSession(str(ensure_pose_weights()), providers=select_providers(ort, device))
        self.device = device_from_providers(self.session.get_providers())
        self.input = self.session.get_inputs()[0].name
        self.batch = batch
        log.info("поза: %s на %s", self.name, self.device)

    @staticmethod
    def _warp(frame: np.ndarray, box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = box
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
        w, h = (x2 - x1) * 1.25, (y2 - y1) * 1.25
        if w > h * W / H:
            h = w * H / W
        else:
            w = h * W / H
        s = np.array([w, h], np.float32)
        m = np.array([[W / w, 0, W / 2 - c[0] * W / w], [0, H / h, H / 2 - c[1] * H / h]], np.float32)
        img = cv2.warpAffine(frame, m, (W, H), flags=cv2.INTER_LINEAR)
        return img, c, s

    def estimate(self, frame: np.ndarray, boxes: Sequence) -> np.ndarray:
        if not len(boxes):
            return np.zeros((0, 17, 3), np.float32)
        out = []
        for i in range(0, len(boxes), self.batch):
            chunk = boxes[i:i + self.batch]
            imgs, cs, ss = [], [], []
            for b in chunk:
                img, c, s = self._warp(frame, b)
                imgs.append(((img[..., ::-1].astype(np.float32) - MEAN) / STD).transpose(2, 0, 1))
                cs.append(c)
                ss.append(s)
            sx, sy = self.session.run(None, {self.input: np.stack(imgs)})
            x = sx.argmax(-1) / 2.0
            y = sy.argmax(-1) / 2.0
            conf = np.minimum(sx.max(-1), sy.max(-1))
            for k in range(len(chunk)):
                kx = x[k] / W * ss[k][0] + cs[k][0] - ss[k][0] / 2
                ky = y[k] / H * ss[k][1] + cs[k][1] - ss[k][1] / 2
                out.append(np.stack([kx, ky, conf[k]], -1))
        return np.asarray(out, np.float32)


def _mean(kps: np.ndarray, idx: Sequence[int], thr: float) -> np.ndarray | None:
    v = [kps[i, :2] for i in idx if kps[i, 2] >= thr]
    return np.mean(v, axis=0) if v else None


def _angle_from_vertical(v: np.ndarray) -> float:
    """0° — вертикально вниз, 90° — горизонтально; вектор «вверх» (dy < 0) — больше 90°."""
    return math.degrees(math.atan2(abs(float(v[0])), float(v[1])))


def posture(kps: np.ndarray | None, box=None, thr: float = 0.3) -> str | None:
    """Сидит / стоит / нагнулся по точкам; None — не понять (нет плеч или бёдер, ноги закрыты столом).

    Камера наблюдения смотрит сверху, поэтому «сидит» — это бедро, повёрнутое к камере или вбок: его проекция
    короткая (< 0,55 длины торса) или почти горизонтальная (> 55° от вертикали). «Нагнулся» — торс наклонён
    больше 50°, а бедро не сидячее. «Стоит» — бедро длинное и почти вертикальное.
    """
    if kps is None:
        return None
    sh = _mean(kps, (L_SH, R_SH), thr)
    hp = _mean(kps, (L_HIP, R_HIP), thr)
    if sh is None or hp is None:
        return None
    torso = hp - sh
    t_len = float(np.hypot(*torso))
    if t_len < 1e-3:
        return None
    t_ang = _angle_from_vertical(torso)
    kn = _mean(kps, (L_KNEE, R_KNEE), thr)
    if kn is not None:
        thigh = kn - hp
        r = float(np.hypot(*thigh)) / t_len
        th_ang = _angle_from_vertical(thigh)
        if r < 0.55 or th_ang > 55:
            return SIT
        if t_ang > 50:
            return BEND
        if r > 0.75 and th_ang < 35:
            return STAND
        return None
    if t_ang > 50:
        return BEND
    return None


def hips_point(kps: np.ndarray | None, thr: float = 0.3) -> tuple[float, float] | None:
    if kps is None:
        return None
    hp = _mean(kps, (L_HIP, R_HIP), thr)
    return None if hp is None else (float(hp[0]), float(hp[1]))
