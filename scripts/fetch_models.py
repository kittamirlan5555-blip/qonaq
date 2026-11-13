"""Скачивает и готовит ONNX-энкодеры для классификаторов ролей/столов в data/weights/encoders/.

Запуск: python scripts/fetch_models.py            # оба энкодера
        python scripts/fetch_models.py mnv2       # только рабочий (выбран в W2, DECISIONS D-017)
Источник — github.com/opencv/opencv_zoo (Apache-2.0). Обрезка классификатора MobileNetV2 требует пакет `onnx`.
Детектор YOLOX качается сам при первом запуске (qonaq/detect/onnx_yolox.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qonaq.roles.encoder import ENCODERS, ensure_encoder  # noqa: E402


def main(argv: list[str]) -> int:
    names = argv or list(ENCODERS)
    bad = 0
    for n in names:
        try:
            p = ensure_encoder(n)
            print(f"ok   {n}: {p} ({p.stat().st_size / 1e6:.1f} МБ, лицензия {ENCODERS[n].license})")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {n}: {e}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
