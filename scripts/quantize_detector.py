"""S2: INT8-квантизация YOLOX для onnxruntime (статическая, QDQ, калибровка на кадрах зала).

    python scripts/quantize_detector.py yolox_s            # -> data/weights/yolox_s_int8.onnx
    python scripts/quantize_detector.py yolox_m --calib data/videos/cctv_restaurant_corner.mp4

Калибровка — 40 кадров ролика (по умолчанию все ролики data/videos). Препроцессинг тот же, что в
qonaq/detect/onnx_yolox.py (letterbox, BGR, 0..255). Голова детектора (последние Concat/Sigmoid) не квантуется —
иначе сильно проседают координаты рамок. Веса — тот же YOLOX (Apache-2.0).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return padded.transpose(2, 0, 1)[None].astype(np.float32)


def calib_frames(videos: list[Path], n: int) -> list[np.ndarray]:
    out = []
    per = max(1, n // max(1, len(videos)))
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for i in np.linspace(0, max(0, total - 1), per).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                out.append(f)
                # и половины кадра — как плитки
                w = f.shape[1]
                out.append(f[:, : int(w * 0.6)])
        cap.release()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", default="yolox_s", nargs="?")
    ap.add_argument("--calib", nargs="*", help="ролики для калибровки (по умолчанию data/videos/*)")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args(argv)
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from qonaq.detect.onnx_yolox import ensure_weights

    src = ensure_weights(args.model)
    model = onnx.load(str(src))
    inp = model.graph.input[0]
    size = int(inp.type.tensor_type.shape.dim[2].dim_value)
    vids = [Path(v) for v in args.calib] if args.calib else sorted(
        p for p in (ROOT / "data" / "videos").glob("*") if p.suffix in (".mp4", ".mov") and "synth" not in p.name)
    frames = calib_frames(vids, args.n)
    print(f"калибровка: {len(frames)} кадров из {len(vids)} роликов, вход {size}×{size}")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{inp.name: letterbox(f, size)} for f in frames])

        def get_next(self):
            return next(self.it, None)

    # веса YOLOX экспортированы в opset 11, а покальная (per-channel) квантизация требует opset ≥ 13
    from onnx import version_converter

    up = src.with_name(f"{src.stem}_op13.onnx")
    if model.opset_import[0].version < 13:
        onnx.save(version_converter.convert_version(model, 13), str(up))
    else:
        onnx.save(model, str(up))
    pre = src.with_name(f"{src.stem}_pre.onnx")
    quant_pre_process(str(up), str(pre), skip_symbolic_shape=True)
    up.unlink(missing_ok=True)
    # не квантуем хвост: декодирование рамок чувствительно к точности
    nodes = onnx.load(str(pre)).graph.node
    tail = [n.name for n in nodes[-12:]]
    dst = src.with_name(f"{src.stem}_int8.onnx")
    quantize_static(str(pre), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, nodes_to_exclude=tail)
    pre.unlink(missing_ok=True)
    print(f"-> {dst} ({dst.stat().st_size / 1e6:.1f} МБ; было {src.stat().st_size / 1e6:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
