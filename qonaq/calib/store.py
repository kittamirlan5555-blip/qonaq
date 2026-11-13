"""Хранилище калибровки зала: data/calib/<site>/ (кропы с блюром, признаки, метки) и models/site/<site>/vN/."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np


def calib_dir(site: str, root: Path | None = None) -> Path:
    from qonaq.config import PROJECT_ROOT

    return (root or PROJECT_ROOT) / "data" / "calib" / site


def models_dir(site: str, root: Path | None = None) -> Path:
    from qonaq.config import PROJECT_ROOT

    return (root or PROJECT_ROOT) / "models" / "site" / site


def load_samples(site: str, root: Path | None = None) -> dict[str, np.ndarray]:
    p = calib_dir(site, root) / "samples.npz"
    if not p.exists():
        raise FileNotFoundError(f"нет {p}: сначала python -m qonaq calibrate collect <видео> --site-id {site}")
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_tracks(site: str, root: Path | None = None) -> list[dict]:
    p = calib_dir(site, root) / "tracks.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def load_labels(site: str, root: Path | None = None) -> dict[str, str]:
    """Ручные метки: key трека -> staff | guest | skip."""
    p = calib_dir(site, root) / "labels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_label(site: str, key: str, label: str, root: Path | None = None) -> dict[str, str]:
    labels = load_labels(site, root)
    labels[key] = label
    p = calib_dir(site, root) / "labels.json"
    p.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return labels


# ------------------------------------------------------------------ версии голов
def versions(site: str, root: Path | None = None) -> list[int]:
    d = models_dir(site, root)
    return sorted(int(p.name[1:]) for p in d.glob("v*") if p.is_dir() and p.name[1:].isdigit())


def new_version_dir(site: str, root: Path | None = None) -> Path:
    v = (versions(site, root) or [0])[-1] + 1
    d = models_dir(site, root) / f"v{v}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _history_path(site: str, root: Path | None) -> Path:
    return models_dir(site, root) / "history.json"


def history(site: str, root: Path | None = None) -> list[dict]:
    p = _history_path(site, root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def active_path(site: str, root: Path | None = None) -> Path:
    """Путь, на который указывает roles.classifier в конфиге объекта."""
    return models_dir(site, root) / "active.npz"


def promote(site: str, version: int | None = None, root: Path | None = None) -> int:
    vs = versions(site, root)
    if not vs:
        raise SystemExit("нет обученных версий: python -m qonaq calibrate train --site-id " + site)
    v = version or vs[-1]
    src = models_dir(site, root) / f"v{v}" / "roles_head.npz"
    if not src.exists():
        raise SystemExit(f"нет {src}")
    shutil.copyfile(src, active_path(site, root))
    h = history(site, root)
    h.append({"action": "promote", "version": v, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _history_path(site, root).write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
    return v


def active_version(site: str, root: Path | None = None) -> int | None:
    cur = None
    for e in history(site, root):
        if e["action"] in ("promote", "rollback"):
            cur = e.get("version")
    return cur


def rollback(site: str, root: Path | None = None) -> int | None:
    """Вернуть предыдущую включённую версию (или выключить голову, если включалась одна)."""
    promoted = [e["version"] for e in history(site, root) if e["action"] in ("promote", "rollback")]
    cur = promoted[-1] if promoted else None
    prev = next((v for v in reversed(promoted[:-1]) if v != cur), None)
    h = history(site, root)
    if prev is None:
        ap = active_path(site, root)
        if ap.exists():
            ap.unlink()
        h.append({"action": "rollback", "version": None, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    else:
        shutil.copyfile(models_dir(site, root) / f"v{prev}" / "roles_head.npz", active_path(site, root))
        h.append({"action": "rollback", "version": prev, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _history_path(site, root).write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
    return prev
