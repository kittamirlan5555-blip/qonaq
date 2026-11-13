"""Скачивает датасеты (Roboflow, CC BY 4.0, формат YOLOv8) из релиза data-v1 в data/datasets/<slug>/.

Запуск: python scripts/fetch_datasets.py            # всё, что нужно для обучения/проверки
        python scripts/fetch_datasets.py --all      # + Table.Cleanliness (не используется в обучении)
        python scripts/fetch_datasets.py --only ramada,restaurant_b

Уже распакованные наборы пропускаются. Работает на Windows и Linux (только stdlib).
Если прямая ссылка релиза недоступна (репозиторий станет приватным), скрипт пробует
`gh api` с заголовком Accept: application/octet-stream (нужен залогиненный gh).
Источники и лицензии — docs/DATA_SOURCES.md.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "kittamirlan5555-blip/qonaq"
TAG = "data-v1"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}/"

# slug -> имя архива в релизе. Slug-и используются в qonaq/datasets/registry.py.
DATASETS: dict[str, str] = {
    "smart_restaurant_a": "AI.Smart.Restaurant.Surveillance.v3i.yolov8.zip",
    "restaurant_b": "restaurant.v3i.yolov8.zip",
    "ramada_c": "Ramada.v3i.yolov8.zip",
    "worker_customer_d": "Worker_Customer_Detection.v5i.yolov8.zip",
    "dirty_table": "Dirty.Table.v1-v1.yolov8.zip",
    "clean_table": "Clean.Table.v4i.yolov8.zip",
    "table_detection": "Table.Detection.v1i.yolov8.zip",
}
OPTIONAL = {"table_cleanliness": "Table.Cleanliness.Detection.v4i.yolov8.zip"}


def _download(name: str, dst: Path) -> None:
    tmp = dst.with_suffix(".part")
    try:
        with urllib.request.urlopen(BASE + name, timeout=60) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.replace(dst)
        return
    except Exception as e:  # noqa: BLE001
        print(f"  прямая ссылка не сработала ({e}); пробую gh api")
    gh = shutil.which("gh")
    if not gh:
        raise RuntimeError("нет ни прямого доступа, ни gh")
    out = subprocess.run(
        [gh, "api", f"repos/{REPO}/releases/tags/{TAG}", "--jq", f'.assets[] | select(.name=="{name}") | .id'],
        check=True, capture_output=True, text=True,
    )
    asset_id = out.stdout.strip()
    with tmp.open("wb") as f:
        subprocess.run([gh, "api", "-H", "Accept: application/octet-stream",
                        f"repos/{REPO}/releases/assets/{asset_id}"], check=True, stdout=f)
    tmp.replace(dst)


def fetch(slug: str, archive: str, root: Path) -> None:
    target = root / slug
    if (target / ".done").exists():
        print(f"skip {slug}")
        return
    zips = root / "_zips"
    zips.mkdir(parents=True, exist_ok=True)
    zpath = zips / archive
    if not zpath.exists():
        print(f"get  {slug} <- {archive}")
        _download(archive, zpath)
    print(f"unzip {slug}")
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target)
    (target / ".done").write_text(archive, encoding="utf-8")
    zpath.unlink()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="включить наборы, не используемые в обучении")
    ap.add_argument("--only", default="", help="через запятую: slug-и")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "data" / "datasets"))
    a = ap.parse_args(argv)
    sets = dict(DATASETS)
    if a.all:
        sets.update(OPTIONAL)
    if a.only:
        wanted = {s.strip() for s in a.only.split(",") if s.strip()}
        sets = {k: v for k, v in {**DATASETS, **OPTIONAL}.items() if k in wanted}
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    failed = []
    for slug, archive in sets.items():
        try:
            fetch(slug, archive, root)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {slug}: {e}")
            failed.append(slug)
    print("done ->", root, "| failed:", failed or "нет")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
