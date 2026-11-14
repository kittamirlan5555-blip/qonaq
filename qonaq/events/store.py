"""SQLite-лог событий и сессий (stdlib sqlite3, без ORM). SPEC §7.

events(id, camera_id, zone_id, ts, type, payload_json)
sessions — сессия стола от посадки до освобождения с вычисленными длительностями.

Сессии строятся из событий (SessionBuilder), а не из видео. Для live-режима есть AsyncEventWriter:
запись идёт в отдельном потоке и не блокирует цикл обработки (SPEC §9a п.6).
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qonaq.logic.base import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    sid INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'table',
    seated_ts REAL NOT NULL,
    served_ts REAL,
    left_ts REAL,
    cleaned_ts REAL,
    response_time_s REAL,
    visit_duration_s REAL,
    cleanup_time_s REAL,
    staff_visits INTEGER NOT NULL DEFAULT 0,
    guests INTEGER,
    outcome TEXT NOT NULL DEFAULT 'open',
    preexisting INTEGER NOT NULL DEFAULT 0,
    dirty_seat INTEGER NOT NULL DEFAULT 0,
    low_confidence INTEGER NOT NULL DEFAULT 0,
    UNIQUE(camera_id, zone_id, sid, kind, seated_ts)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alerts_sent (
    alert_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL, zone_id TEXT NOT NULL, kind TEXT NOT NULL, ts REAL NOT NULL,
    text TEXT NOT NULL DEFAULT '', confident INTEGER NOT NULL DEFAULT 1, sent INTEGER NOT NULL DEFAULT 0,
    wall REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL, verdict TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'telegram',
    user TEXT NOT NULL DEFAULT '', wall REAL NOT NULL, note TEXT NOT NULL DEFAULT ''
);
"""

SESSION_COLS = (
    "camera_id", "zone_id", "sid", "kind", "seated_ts", "served_ts", "left_ts", "cleaned_ts",
    "response_time_s", "visit_duration_s", "cleanup_time_s", "staff_visits", "guests", "outcome",
    "preexisting", "dirty_seat", "low_confidence",
)


class EventStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._migrate_sessions()
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._open: dict[tuple[str, str, str], dict[str, Any]] = {}  # незакрытые сессии
        self._dirty_next: set[tuple[str, str, str]] = set()  # следующая посадка — за неубранный стол

    def _migrate_sessions(self) -> None:
        """Старые журналы: сессия была уникальна по (камера, стол, sid). После перезапуска службы sid начинаются
        заново и новые посадки затирали старые (S5). Теперь в ключе ещё и seated_ts; таблица пересобирается."""
        row = self._db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()
        if row is None or "kind, seated_ts)" in row[0]:
            return
        with self._db:
            self._db.execute("ALTER TABLE sessions RENAME TO sessions_old")
            self._db.executescript(SCHEMA)
            cols = ",".join(SESSION_COLS)
            self._db.execute(f"INSERT OR IGNORE INTO sessions({cols}) SELECT {cols} FROM sessions_old")
            self._db.execute("DROP TABLE sessions_old")

    # ------------------------------------------------------------------ meta
    def set_meta(self, **kv: Any) -> None:
        with self._lock, self._db:
            for k, v in kv.items():
                self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (k, json.dumps(v)))

    def meta(self) -> dict[str, Any]:
        with self._lock:
            return {r["key"]: json.loads(r["value"]) for r in self._db.execute("SELECT key, value FROM meta")}

    # ------------------------------------------------------------------ write
    def add(self, events: Iterable[Event]) -> None:
        events = list(events)
        if not events:
            return
        with self._lock, self._db:
            for e in events:
                self._db.execute(
                    "INSERT INTO events(camera_id, zone_id, ts, type, payload_json) VALUES (?,?,?,?,?)",
                    (e.camera_id, e.zone_id, float(e.ts), e.type, json.dumps(e.payload, ensure_ascii=False)),
                )
                self._apply_to_session(e)

    def _save_session(self, s: dict[str, Any]) -> None:
        cols = ",".join(SESSION_COLS)
        ph = ",".join("?" for _ in SESSION_COLS)
        self._db.execute(f"INSERT OR REPLACE INTO sessions({cols}) VALUES ({ph})",
                         [s.get(c) for c in SESSION_COLS])

    def _apply_to_session(self, e: Event) -> None:
        """Сессии столов (restaurant) и ожидания (clinic) — производные от событий."""
        p = e.payload
        t = e.type
        if t in ("wait_start", "wait_end", "service_start", "service_end"):
            return self._apply_clinic(e)
        key = (e.camera_id, e.zone_id, "table")
        s = self._open.get(key)
        if t == "seated_at_dirty_table":
            self._dirty_next.add(key)
            return
        if t == "seated":
            s = {
                "camera_id": e.camera_id, "zone_id": e.zone_id, "sid": int(p.get("sid", 0)), "kind": "table",
                "seated_ts": e.ts, "staff_visits": 0, "guests": p.get("guests"), "outcome": "open",
                "preexisting": int(bool(p.get("preexisting"))),
                "dirty_seat": int(key in self._dirty_next),
                "low_confidence": int(p.get("confidence") == "low"),
            }
            self._dirty_next.discard(key)
            if p.get("preexisting"):
                s["served_ts"] = e.ts
            self._open[key] = s
        elif s is None:
            return
        elif t == "served":
            s["served_ts"] = e.ts
            s["response_time_s"] = p.get("response_time_s")
            s["staff_visits"] = max(s["staff_visits"], 1)
            s["low_confidence"] = int(s["low_confidence"] or p.get("confidence") == "low")
        elif t == "staff_visit":
            s["staff_visits"] = int(p.get("n", s["staff_visits"] + 1))
        elif t == "left":
            s["left_ts"] = e.ts
            s["visit_duration_s"] = p.get("visit_duration_s")
            s["outcome"] = "left"
        elif t == "walkout_before_service":
            s["left_ts"] = e.ts
            s["visit_duration_s"] = p.get("waited_s")
            s["outcome"] = "walkout"
            self._save_session(s)
            del self._open[key]
            return
        elif t == "cleaned":
            s["cleaned_ts"] = e.ts
            s["cleanup_time_s"] = p.get("cleanup_time_s")
            s["outcome"] = "completed"
            self._save_session(s)
            del self._open[key]
            return
        else:
            return
        self._save_session(s)

    def _apply_clinic(self, e: Event) -> None:
        p = e.payload
        kind = "wait" if e.type.startswith("wait") else "service"
        key = (e.camera_id, e.zone_id, f"{kind}:{p.get('sid')}")
        if e.type.endswith("_start"):
            self._open[key] = {
                "camera_id": e.camera_id, "zone_id": e.zone_id, "sid": int(p.get("sid", 0)), "kind": kind,
                "seated_ts": e.ts, "staff_visits": 0, "outcome": "open", "preexisting":
                int(bool(p.get("preexisting"))), "dirty_seat": 0, "low_confidence": 0,
            }
            self._save_session(self._open[key])
            return
        s = self._open.pop(key, None)
        if s is None:
            return
        s["left_ts"] = e.ts
        s["visit_duration_s"] = p.get("duration_s")
        s["outcome"] = p.get("outcome", "completed")
        if kind == "wait" and s["outcome"] == "served":
            s["served_ts"] = e.ts
            s["response_time_s"] = p.get("duration_s")
        self._save_session(s)

    # ------------------------------------------------------------------ read
    def events(self, since_id: int = 0, types: Iterable[str] | None = None, limit: int | None = None,
               newest_first: bool = False) -> list[dict[str, Any]]:
        q = "SELECT * FROM events WHERE id > ?"
        args: list[Any] = [since_id]
        if types:
            types = list(types)
            q += f" AND type IN ({','.join('?' for _ in types)})"
            args += types
        q += " ORDER BY id DESC" if newest_first else " ORDER BY ts, id"
        if limit:
            q += " LIMIT ?"
            args.append(limit)
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [
            {"id": r["id"], "camera_id": r["camera_id"], "zone_id": r["zone_id"], "ts": r["ts"],
             "type": r["type"], "payload": json.loads(r["payload_json"])}
            for r in rows
        ]

    def sessions(self, kind: str | None = "table") -> list[dict[str, Any]]:
        q = "SELECT * FROM sessions"
        args: list[Any] = []
        if kind:
            q += " WHERE kind = ?"
            args.append(kind)
        with self._lock:
            return [dict(r) for r in self._db.execute(q + " ORDER BY seated_ts, id", args)]

    # ------------------------------------------------------------------ алерты и обратная связь (W7/W8)
    def add_alert(self, alert_id: str, camera_id: str, zone_id: str, kind: str, ts: float, text: str,
                  confident: bool, sent: bool, wall: float) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO alerts_sent VALUES (?,?,?,?,?,?,?,?,?)",
                             (alert_id, camera_id, zone_id, kind, float(ts), text, int(confident), int(sent), wall))

    def alerts_sent(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM alerts_sent ORDER BY ts").fetchall()
        return [dict(r) for r in rows]

    def add_feedback(self, alert_id: str, verdict: str, source: str = "telegram", user: str = "",
                     wall: float | None = None, note: str = "") -> None:
        import time as _t

        with self._lock, self._db:
            self._db.execute("INSERT INTO feedback(alert_id, verdict, source, user, wall, note) VALUES (?,?,?,?,?,?)",
                             (alert_id, verdict, source, user, wall or _t.time(), note))

    def feedback(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT f.*, a.camera_id, a.zone_id, a.kind, a.ts FROM feedback f "
                "LEFT JOIN alerts_sent a ON a.alert_id = f.alert_id ORDER BY f.id").fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.close()


class AsyncEventWriter:
    """Очередь + поток записи: цикл обработки кадров не ждёт диск."""

    def __init__(self, store: EventStore, on_events=None):
        self.store = store
        self._q: queue.Queue = queue.Queue()
        self._on_events = on_events
        self._thread = threading.Thread(target=self._run, name="event-writer", daemon=True)
        self._thread.start()

    def add(self, events: Iterable[Event]) -> None:
        events = list(events)
        if events:
            self._q.put(events)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                self.store.add(item)
                if self._on_events:
                    self._on_events(item)
            except Exception:  # noqa: BLE001
                import logging

                logging.getLogger("qonaq.events").exception("ошибка записи событий")

    def pending(self) -> int:
        return self._q.qsize()

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)


def open_store(run_dir: str | Path) -> EventStore:
    return EventStore(Path(run_dir) / "events.sqlite")
