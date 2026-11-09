"""Модели зон: полигон в координатах кадра камеры + тип + id + имя (SPEC §3)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ZoneType = Literal["table", "entrance", "waiting_area", "service_point", "ignore"]
AnchorStrategy = Literal["bottom_center", "center", "overlap", "hips"]

ZONE_TYPES: tuple[str, ...] = ("table", "entrance", "waiting_area", "service_point", "ignore")
ID_PREFIX = {"table": "T", "entrance": "E", "waiting_area": "W", "service_point": "S", "ignore": "X"}


def default_anchor(zone_type: str) -> str:
    """SPEC §9a п.2: overlap для столов, bottom_center для проходных зон."""
    return "overlap" if zone_type == "table" else "bottom_center"


class Zone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: ZoneType = "table"
    name: str = ""
    polygon: list[tuple[float, float]]
    anchor: AnchorStrategy | None = None  # None -> по типу зоны
    overlap_threshold: float = Field(0.3, ge=0.0, le=1.0)
    seats: int | None = None
    # service_point: полигон «за стойкой», где стоит сотрудник (для алерта «стойка пустует»).
    staff_polygon: list[tuple[float, float]] | None = None

    @field_validator("polygon")
    @classmethod
    def _min_points(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("полигон зоны должен содержать минимум 3 точки")
        return v

    @property
    def anchor_strategy(self) -> str:
        return self.anchor or default_anchor(self.type)

    @property
    def title(self) -> str:
        return self.name or self.id

    def np_polygon(self) -> np.ndarray:
        return np.array(self.polygon, dtype=np.float32)

    def centroid(self) -> tuple[float, float]:
        p = self.np_polygon()
        return float(p[:, 0].mean()), float(p[:, 1].mean())

    def bbox(self) -> tuple[float, float, float, float]:
        p = self.np_polygon()
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


class ZoneSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = ""
    frame_size: tuple[int, int] | None = None  # (width, height) кадра, на котором размечали
    zones: list[Zone] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> ZoneSet:
        ids = [z.id for z in self.zones]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"повторяющиеся id зон: {sorted(dup)}")
        return self

    def by_type(self, *types: str) -> list[Zone]:
        return [z for z in self.zones if z.type in types]

    def get(self, zone_id: str) -> Zone:
        for z in self.zones:
            if z.id == zone_id:
                return z
        raise KeyError(zone_id)

    def next_id(self, zone_type: str) -> str:
        prefix = ID_PREFIX[zone_type]
        n = 1
        existing = {z.id for z in self.zones}
        while f"{prefix}{n:02d}" in existing:
            n += 1
        return f"{prefix}{n:02d}"

    def scaled(self, width: int, height: int) -> ZoneSet:
        """Пересчёт полигонов, если кадр источника другого размера, чем при разметке."""
        if not self.frame_size or tuple(self.frame_size) == (width, height):
            return self
        sx, sy = width / self.frame_size[0], height / self.frame_size[1]

        def sc(poly):
            return [(x * sx, y * sy) for x, y in poly] if poly else poly

        zones = [
            z.model_copy(update={"polygon": sc(z.polygon), "staff_polygon": sc(z.staff_polygon)})
            for z in self.zones
        ]
        return ZoneSet(camera_id=self.camera_id, frame_size=(width, height), zones=zones)


def load_zones(path: str | Path) -> ZoneSet:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ZoneSet.model_validate(data)


def save_zones(zs: ZoneSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = zs.model_dump(exclude_none=True)
    if zs.frame_size:
        data["frame_size"] = list(zs.frame_size)
    for z in data["zones"]:
        z["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in z["polygon"]]
        if z.get("staff_polygon"):
            z["staff_polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in z["staff_polygon"]]
        if z.get("overlap_threshold") == 0.3:
            z.pop("overlap_threshold")
        if not z.get("name"):
            z.pop("name", None)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=None, width=120),
        encoding="utf-8",
    )
    return path
