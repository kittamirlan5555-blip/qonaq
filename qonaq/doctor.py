"""`python -m qonaq doctor` — проверка окружения: Python, torch/CUDA, детектор, видео, веса."""
from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
from pathlib import Path

from qonaq.config import PROJECT_ROOT

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _version(mod: str) -> str | None:
    if importlib.util.find_spec(mod) is None:
        return None
    try:
        m = importlib.import_module(mod)
        return str(getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        return f"ошибка импорта: {e}"


def list_videos(root: Path | None = None) -> list[Path]:
    d = (root or PROJECT_ROOT) / "data" / "videos"
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXT)


def collect() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "packages": {m: _version(m) for m in ("numpy", "cv2", "supervision", "pydantic", "fastapi",
                                               "onnxruntime", "torch", "ultralytics")},
        "cuda": False, "gpu": None, "device": "cpu", "detector": None, "detector_error": None,
        "videos": [], "weights": [], "ffmpeg": None, "telegram": False, "warnings": [],
    }
    if info["packages"]["torch"] and "ошибка" not in info["packages"]["torch"]:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["device"] = "cuda:0"
    if info["packages"]["onnxruntime"]:
        import onnxruntime as ort

        info["onnx_providers"] = ort.get_available_providers()
    v = sys.version_info
    if not (3, 11) <= (v.major, v.minor) <= (3, 13):
        info["warnings"].append("нужен Python 3.11–3.12")
    wd = PROJECT_ROOT / "data" / "weights"
    if wd.exists():
        info["weights"] = sorted(f"{p.name} ({p.stat().st_size / 1e6:.0f} МБ)" for p in wd.iterdir()
                                 if p.suffix in (".pt", ".onnx"))
    from qonaq.sources.video import probe

    for p in list_videos():
        try:
            i = probe(p)
            info["videos"].append({"name": p.name, "duration_s": round(i.duration_s, 1), "fps": round(i.fps, 2),
                                   "size": f"{i.width}x{i.height}"})
        except OSError as e:
            info["videos"].append({"name": p.name, "error": str(e)})
    try:
        from qonaq.render.sink import find_ffmpeg

        info["ffmpeg"] = find_ffmpeg()
    except Exception:  # noqa: BLE001
        pass
    import os

    from qonaq.config import load_dotenv_if_any

    load_dotenv_if_any(PROJECT_ROOT)
    info["telegram"] = bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))
    return info


def check_detector(info: dict) -> None:
    """Пробует создать детектор по умолчанию и прогнать один чёрный кадр."""
    import numpy as np

    from qonaq.config import DetectorConfig
    from qonaq.detect.base import create_detector

    try:
        det = create_detector(DetectorConfig())
        det.detect([np.zeros((360, 640, 3), np.uint8)])
        info["detector"] = f"{det.name} ({type(det).__name__}), устройство: {det.device}"
        info["device"] = det.device
    except Exception as e:  # noqa: BLE001
        info["detector_error"] = f"{type(e).__name__}: {e}"


def run(check_model: bool = True) -> int:
    info = collect()
    if check_model:
        check_detector(info)
    ok = lambda b: "OK " if b else "-- "  # noqa: E731
    print("Qonaq Vision — проверка окружения\n")
    print(f"  Python      {info['python']}   {info['platform']}")
    for name, ver in info["packages"].items():
        print(f"  {ok(ver)}{name:<12} {ver or 'не установлен'}")
    # CUDA через PyTorch; детектор onnxruntime может быть на GPU и без него — см. строку «детектор» ниже
    print(f"  CUDA torch  {'да: ' + str(info['gpu']) if info['cuda'] else 'нет (для onnxruntime не нужен)'}")
    if info.get("onnx_providers"):
        print(f"  ONNX        {', '.join(info['onnx_providers'])}")
    print(f"  ffmpeg      {info['ffmpeg'] or 'не найден (видео будет в mp4v, не H.264)'}")
    print(f"  Telegram    {'токен найден в .env' if info['telegram'] else 'не настроен (алерты пишутся в лог)'}")
    weights = ", ".join(info["weights"]) or "нет — скачаются при первом запуске"
    print(f"\n  Веса модели ({len(info['weights'])}): {weights}")
    if check_model:
        if info["detector"]:
            print(f"  OK детектор: {info['detector']}")
        else:
            print(f"  !! детектор не запустился: {info['detector_error']}")
    print(f"\n  Видео в data/videos ({len(info['videos'])}):")
    for v in info["videos"]:
        if "error" in v:
            print(f"    !! {v['name']}: {v['error']}")
        else:
            print(f"    {v['name']:<34} {v['duration_s']:>6.1f} с  {v['fps']:>6.2f} fps  {v['size']}")
    if not info["videos"]:
        print("    нет — запустите: python scripts/fetch_videos.py")
    for w in info["warnings"]:
        print(f"\n  ВНИМАНИЕ: {w}")
    failed = check_model and not info["detector"]
    print("\nИтог:", "есть проблемы (см. выше)" if failed else "всё готово к работе")
    return 1 if failed else 0
