"""Ресурсы процесса без лишних зависимостей: RSS-память и CPU (S1, долгие прогоны и утечки).

psutil используется, если установлен; иначе — /proc (Linux), WinAPI через ctypes (Windows), getrusage (macOS,
там это пиковая, а не текущая память — помечено в `rss_kind`).
"""
from __future__ import annotations

import os
import sys
import time


def rss_mb() -> tuple[float | None, str]:
    """(RSS в МБ, вид): вид = current | peak | none."""
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.Process().memory_info().rss / 2**20, "current"
    except Exception:  # noqa: BLE001
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="ascii") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024, "current"
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / 2**20, "current"
        except Exception:  # noqa: BLE001
            pass
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (r / 2**20 if sys.platform == "darwin" else r / 1024), "peak"
    except Exception:  # noqa: BLE001
        return None, "none"


class CpuMeter:
    """Загрузка CPU процессом между вызовами, в % одного ядра (200% = два ядра заняты полностью)."""

    def __init__(self) -> None:
        self._t = time.perf_counter()
        self._c = time.process_time()

    def sample(self) -> float:
        t, c = time.perf_counter(), time.process_time()
        dt = max(t - self._t, 1e-6)
        pct = 100.0 * (c - self._c) / dt
        self._t, self._c = t, c
        return round(pct, 1)


def cpu_count() -> int:
    return os.cpu_count() or 1
