"""Реестр датасетов и единая схема классов (WEEKEND_PLAN W1)."""
from __future__ import annotations

from dataclasses import dataclass, field

ROLE_LABELS = ("guest", "staff")
TABLE_LABELS = ("clean", "dirty", "occupied")


@dataclass(frozen=True)
class DatasetSpec:
    slug: str            # папка в data/datasets/
    kind: str            # roles | tables
    venue: str           # A/B/C/D для ролей; для столов — имя набора (камеры выделяются по имени файла)
    class_map: dict[str, str] = field(default_factory=dict)  # исходное имя класса -> единое; нет в словаре = игнор
    train_ok: bool = True  # False — только внешний тест
    note: str = ""


_STAFF = "staff"
_GUEST = "guest"

DATASETS: dict[str, DatasetSpec] = {
    "smart_restaurant_a": DatasetSpec(
        "smart_restaurant_a", "roles", "A",
        {"customer": _GUEST, "waiter": _STAFF, "staff": _STAFF},
        note="одна потолочная fisheye-камера; класс person игнорируется",
    ),
    "restaurant_b": DatasetSpec("restaurant_b", "roles", "B", {"Customer": _GUEST, "Staff": _STAFF},
                                note="CCTV над входом кафе"),
    "ramada_c": DatasetSpec("ramada_c", "roles", "C", {"Waiter": _STAFF},
                            note="гости не размечены — добираются детектором (IoU<0.3 с Waiter)"),
    "worker_customer_d": DatasetSpec("worker_customer_d", "roles", "D", {"Customer": _GUEST, "Worker": _STAFF},
                                     note="офис/класс, не ресторан — слабый зал"),
    "dirty_table": DatasetSpec("dirty_table", "tables", "dirty_table",
                               {"unoccupied_clean": "clean", "unoccupied_dirty": "dirty"}),
    "clean_table": DatasetSpec("clean_table", "tables", "clean_table",
                               {"occupied": "occupied", "unoccupied_clean": "clean", "unoccupied_dirty": "dirty"}),
    "table_detection": DatasetSpec("table_detection", "tables", "table_detection",
                                   {"dirty-tables": "dirty", "occupied": "occupied", "unoccupied-clean": "clean"},
                                   train_ok=False, note="69 кадров — только внешний тест"),
}

# Главный показатель ролей — среднее по этим залам (D — слабый, не ресторан).
MAIN_VENUES = ("A", "B", "C")


def role_specs() -> list[DatasetSpec]:
    return [s for s in DATASETS.values() if s.kind == "roles"]


def table_specs() -> list[DatasetSpec]:
    return [s for s in DATASETS.values() if s.kind == "tables"]
