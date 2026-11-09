"""POSAdapter: что нам нужно от кассовой системы — открытые чеки с привязкой к столам.

Сценарий Phase 2 (BUSINESS.md п.7, анти-фрод): стол занят N минут, а чек в POS не открыт -> алерт
`occupied_without_check`. Логика стола вызывает только `has_open_check(zone_id, ts)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml


@dataclass(frozen=True)
class Check:
    table: str  # номер/имя стола в POS
    opened_ts: float
    closed_ts: float | None = None
    total: float | None = None

    def is_open(self, ts: float) -> bool:
        return self.opened_ts <= ts and (self.closed_ts is None or ts < self.closed_ts)


@runtime_checkable
class POSAdapter(Protocol):
    def open_checks(self, ts: float) -> list[Check]:
        """Чеки, открытые на момент ts."""
        ...

    def has_open_check(self, zone_id: str, ts: float) -> bool: ...


class BasePOS:
    """Общая часть: сопоставление зон (T03) и столов POS («3», «Зал-3»)."""

    def __init__(self, table_map: dict[str, str] | None = None):
        self.table_map = dict(table_map or {})

    def pos_table(self, zone_id: str) -> str:
        return self.table_map.get(zone_id, zone_id)

    def open_checks(self, ts: float) -> list[Check]:  # pragma: no cover — интерфейс
        raise NotImplementedError

    def has_open_check(self, zone_id: str, ts: float) -> bool:
        table = self.pos_table(zone_id)
        return any(c.table == table for c in self.open_checks(ts))


class MockPOS(BasePOS):
    """Чеки по сценарию — для тестов и демо без настоящей кассы.

    Сценарий YAML:
        checks:
          - {table: T01, opened: 120, closed: 2400, total: 18500}
          - {table: T02, opened: 300}
    """

    def __init__(self, checks: list[Check] | None = None, table_map: dict[str, str] | None = None):
        super().__init__(table_map)
        self.checks = list(checks or [])

    @classmethod
    def from_yaml(cls, path: str | Path, table_map: dict[str, str] | None = None) -> MockPOS:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        checks = [Check(str(c["table"]), float(c["opened"]), float(c["closed"]) if c.get("closed") is not None
                        else None, c.get("total")) for c in data.get("checks", [])]
        return cls(checks, table_map)

    def open_check(self, table: str, ts: float) -> None:
        self.checks.append(Check(table, ts))

    def open_checks(self, ts: float) -> list[Check]:
        return [c for c in self.checks if c.is_open(ts)]


def create_pos(cfg) -> POSAdapter | None:
    """Фабрика по `SiteConfig.pos`."""
    if cfg.adapter == "none":
        return None
    if cfg.adapter == "mock":
        if cfg.scenario:
            from qonaq.config import resolve_path

            return MockPOS.from_yaml(resolve_path(cfg.scenario), cfg.table_map)
        return MockPOS(table_map=cfg.table_map)
    from qonaq.pos import stubs

    return {"iiko": stubs.IikoPOS, "rkeeper": stubs.RKeeperPOS, "poster": stubs.PosterPOS}[cfg.adapter](cfg.table_map)
