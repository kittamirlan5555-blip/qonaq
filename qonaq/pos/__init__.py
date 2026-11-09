"""POS-интеграция (Phase 2): интерфейс POSAdapter, MockPOS и заглушки iiko / r_keeper / Poster."""
from qonaq.pos.base import Check, MockPOS, POSAdapter, create_pos

__all__ = ["Check", "MockPOS", "POSAdapter", "create_pos"]
