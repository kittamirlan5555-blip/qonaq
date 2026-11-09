"""Алерты: Telegram Bot API через обычный HTTP (httpx). Включается только при наличии
TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в .env; без них алерты пишутся в лог.

Очередь + ретраи с экспоненциальной задержкой (интернет в заведениях нестабилен) + дедупликация:
один алерт на событие (камера, зона, тип, сессия).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from qonaq.logic.base import Event
from qonaq.strings import alert_text

log = logging.getLogger("qonaq.alerts")


class Sender(Protocol):
    def send(self, text: str) -> None: ...  # бросает исключение при неудаче


class TelegramSender:
    def __init__(self, token: str, chat_id: str, client=None, timeout: float = 10.0):
        import httpx

        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id
        self._client = client or httpx.Client(timeout=timeout)

    def send(self, text: str) -> None:
        r = self._client.post(self._url, json={"chat_id": self.chat_id, "text": text,
                                               "disable_web_page_preview": True})
        if r.status_code != 200:
            raise RuntimeError(f"Telegram API {r.status_code}: {r.text[:200]}")


def sender_from_env() -> TelegramSender | None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        return TelegramSender(token, chat)
    return None


def alert_key(e: Event) -> tuple:
    p = e.payload
    return (e.camera_id, e.zone_id, p.get("kind"), p.get("sid"), p.get("seq"))


def alert_seconds(payload: dict[str, Any]) -> float:
    for k in ("waited_s", "dirty_s", "since_visit_s", "occupied_s", "unattended_s", "wait_s"):
        if k in payload:
            return float(payload[k])
    return 0.0


def format_alert(e: Event, site_name: str = "", zone_title: str | None = None) -> str:
    text = alert_text(e.payload.get("kind", ""), zone_title or e.zone_id, alert_seconds(e.payload))
    return f"⚠ {site_name}: {text}" if site_name else f"⚠ {text}"


@dataclass
class AlertRecord:
    ts: float
    camera_id: str
    zone_id: str
    kind: str
    text: str
    sent: bool = False
    wall: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "camera_id": self.camera_id, "zone_id": self.zone_id, "kind": self.kind,
                "text": self.text, "sent": self.sent, "wall": self.wall}


class AlertDispatcher:
    def __init__(self, sender: Sender | None = None, site_name: str = "", max_queue: int = 500,
                 retry_base_s: float = 2.0, retry_max_s: float = 120.0, sleep=None):
        self.sender = sender
        self.site_name = site_name
        self.feed: deque[AlertRecord] = deque(maxlen=200)
        self._seen: set[tuple] = set()
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._retry_base, self._retry_max = retry_base_s, retry_max_s
        self._stop = threading.Event()
        self._sleep = sleep or (lambda s: self._stop.wait(s))
        self._thread: threading.Thread | None = None
        self.sent_count = 0
        self.failed_attempts = 0
        if sender is not None:
            self._thread = threading.Thread(target=self._run, name="alerts", daemon=True)
            self._thread.start()
        else:
            log.info("Telegram не настроен (нет TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID) — алерты только в лог и панель")

    def handle(self, events: list[Event], zone_titles: dict[str, str] | None = None) -> list[AlertRecord]:
        """Принимает события, выбирает алерты, дедуплицирует, ставит в очередь отправки."""
        out = []
        for e in events:
            if e.type != "alert":
                continue
            key = alert_key(e)
            if key in self._seen:
                continue
            self._seen.add(key)
            rec = AlertRecord(e.ts, e.camera_id, e.zone_id, e.payload.get("kind", ""),
                              format_alert(e, self.site_name, (zone_titles or {}).get(e.zone_id)))
            self.feed.append(rec)
            out.append(rec)
            log.warning("АЛЕРТ %s", rec.text)
            if self.sender is not None:
                try:
                    self._q.put_nowait(rec)
                except queue.Full:
                    log.error("очередь алертов переполнена — алерт не отправлен: %s", rec.text)
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rec = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            delay = self._retry_base
            while not self._stop.is_set():
                try:
                    self.sender.send(rec.text)
                    rec.sent = True
                    self.sent_count += 1
                    break
                except Exception as ex:  # noqa: BLE001 — сеть/Telegram недоступны: ждём и повторяем
                    self.failed_attempts += 1
                    log.warning("алерт не отправлен (%s), повтор через %.0f с", ex, delay)
                    self._sleep(delay)
                    delay = min(delay * 2, self._retry_max)

    def pending(self) -> int:
        return self._q.qsize()

    def close(self, drain_s: float = 0.0) -> None:
        deadline = time.time() + drain_s
        while drain_s and self._q.qsize() and time.time() < deadline:
            time.sleep(0.05)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
