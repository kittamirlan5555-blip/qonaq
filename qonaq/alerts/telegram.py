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
from qonaq.strings import alert_message, t

log = logging.getLogger("qonaq.alerts")


class Sender(Protocol):
    def send(self, text: str) -> None: ...  # бросает исключение при неудаче


DEFAULT_API_BASE = "https://api.telegram.org"


def api_base() -> str:
    """Адрес Bot API: TELEGRAM_API_BASE (локальный фейковый сервер для проверки, свой прокси) или официальный."""
    return (os.environ.get("TELEGRAM_API_BASE") or DEFAULT_API_BASE).rstrip("/")


class TelegramSender:
    def __init__(self, token: str, chat_id: str, client=None, timeout: float = 10.0, base_url: str | None = None):
        import httpx

        self.base_url = (base_url or api_base()).rstrip("/")
        self._base = f"{self.base_url}/bot{token}/"
        self._url = self._base + "sendMessage"
        self.chat_id = chat_id
        self._client = client or httpx.Client(timeout=timeout)

    def send(self, text: str, feedback_id: str | None = None) -> None:
        """feedback_id — под сообщением кнопка «Неверно» (callback `fb:<id>`, ловит FeedbackPoller)."""
        body: dict[str, Any] = {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True}
        if feedback_id:
            body["reply_markup"] = {"inline_keyboard": [[{"text": "Неверно", "callback_data": f"fb:{feedback_id}"}]]}
        r = self._client.post(self._url, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram API {r.status_code}: {r.text[:200]}")

    def send_photo(self, data: bytes, caption: str = "") -> None:
        """Один JPEG с подписью (sendPhoto)."""
        r = self._client.post(self._base + "sendPhoto", data={"chat_id": self.chat_id, "caption": caption[:1000]},
                              files={"photo": ("frame.jpg", data, "image/jpeg")})
        if r.status_code != 200:
            raise RuntimeError(f"Telegram API {r.status_code}: {r.text[:200]}")

    def send_photos(self, photos: list[tuple[bytes, str]]) -> None:
        """До 10 JPEG с подписями одним альбомом (sendMediaGroup); один кадр — sendPhoto."""
        import json as _json

        if not photos:
            return
        if len(photos) == 1:
            self.send_photo(*photos[0])
            return
        media, files = [], {}
        for i, (data, caption) in enumerate(photos[:10]):
            media.append({"type": "photo", "media": f"attach://p{i}", "caption": caption[:1000]})
            files[f"p{i}"] = (f"p{i}.jpg", data, "image/jpeg")
        r = self._client.post(self._base + "sendMediaGroup", data={"chat_id": self.chat_id,
                                                                  "media": _json.dumps(media, ensure_ascii=False)},
                              files=files)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram API {r.status_code}: {r.text[:200]}")

    def call(self, method: str, **body) -> dict:
        kw: dict[str, Any] = {}
        if body.get("timeout"):  # long polling getUpdates: ответ придёт не раньше timeout — ждём дольше
            kw["timeout"] = float(body["timeout"]) + 10.0
        r = self._client.post(self._base + method, json=body, **kw)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram API {r.status_code}: {r.text[:200]}")
        return r.json()


def sender_from_env(chat_env: str = "TELEGRAM_CHAT_ID") -> TelegramSender | None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get(chat_env)
    if token and chat:
        return TelegramSender(token, chat)
    return None


class FeedbackPoller:
    """Кнопка «Неверно» под алертом → запись в таблицу feedback (пригодится калибровке, W6/W7).
    Long polling getUpdates в отдельном потоке; без вебхука и без открытых портов."""

    def __init__(self, sender: TelegramSender, store, timeout_s: int = 20):
        self.sender, self.store, self.timeout_s = sender, store, timeout_s
        self._offset = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def skip_pending(self) -> int:
        """Пропустить уже накопившиеся обновления (старые нажатия, сообщения «привет» боту): дальше — только новые."""
        res = self.sender.call("getUpdates", offset=-1, timeout=0)
        ids = [int(u["update_id"]) for u in res.get("result", [])]
        if ids:
            self._offset = max(ids) + 1
        return len(ids)

    def poll_once(self) -> int:
        res = self.sender.call("getUpdates", offset=self._offset, timeout=self.timeout_s,
                               allowed_updates=["callback_query"])
        n = 0
        for u in res.get("result", []):
            self._offset = max(self._offset, int(u["update_id"]) + 1)
            cq = u.get("callback_query")
            if not cq or not str(cq.get("data", "")).startswith("fb:"):
                continue
            alert_id = cq["data"][3:]
            user = (cq.get("from") or {}).get("username") or str((cq.get("from") or {}).get("id", ""))
            self.store.add_feedback(alert_id, "wrong", source="telegram", user=user)
            try:
                self.sender.call("answerCallbackQuery", callback_query_id=cq["id"], text="Записано: неверно. Спасибо!")
            except Exception as e:  # noqa: BLE001
                log.debug("answerCallbackQuery: %s", e)
            n += 1
        return n

    def start(self) -> FeedbackPoller:
        def run():
            delay = 2.0
            while not self._stop.is_set():
                try:
                    self.poll_once()
                    delay = 2.0
                except Exception as e:  # noqa: BLE001
                    log.debug("getUpdates: %s", e)
                    self._stop.wait(delay)
                    delay = min(delay * 2, 120)

        self._thread = threading.Thread(target=run, name="tg-feedback", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


def alert_key(e: Event) -> tuple:
    p = e.payload
    return (e.camera_id, e.zone_id, p.get("kind"), p.get("sid"), p.get("seq"))


def alert_seconds(payload: dict[str, Any]) -> float:
    for k in ("waited_s", "dirty_s", "since_visit_s", "occupied_s", "unattended_s", "wait_s"):
        if k in payload:
            return float(payload[k])
    return 0.0


def format_alert(e: Event, site_name: str = "", zone_title: str | None = None, wall: float | None = None,
                 tz: str | None = None) -> str:
    """Текст алерта администратору (S4): время, номер стола, что случилось — без технических слов.

    site_name добавляется второй строкой только если задан (у одного администратора несколько заведений)."""
    text = alert_message(e.payload.get("kind", ""), zone_title or e.zone_id, alert_seconds(e.payload), wall, tz)
    return f"{text}\n{site_name}" if site_name else text


CONFIDENCE_KINDS = ("no_waiter", "long_service_without_visit", "dirty_table")


def is_confident(e: Event, min_conf: float) -> bool:
    """W8: алерт о гостях уверенный, если гости подтверждены ролью и уверенность ролей ≥ min_conf.
    Нет полей уверенности (clinic, старые события) — считается уверенным."""
    if e.payload.get("kind") not in CONFIDENCE_KINDS:
        return True
    if not e.payload.get("by_role", True):
        return False
    c = e.payload.get("confidence")
    return c is None or float(c) >= min_conf


def alert_id(e: Event) -> str:
    """Короткий id для кнопки «Неверно» (callback_data ≤ 64 байт)."""
    import hashlib

    return hashlib.blake2b(repr((alert_key(e), round(e.ts, 1))).encode(), digest_size=8).hexdigest()


@dataclass
class AlertRecord:
    ts: float
    camera_id: str
    zone_id: str
    kind: str
    text: str
    sent: bool = False
    wall: float = field(default_factory=time.time)
    confident: bool = True
    alert_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "camera_id": self.camera_id, "zone_id": self.zone_id, "kind": self.kind,
                "text": self.text, "sent": self.sent, "wall": self.wall, "confident": self.confident,
                "alert_id": self.alert_id}


class AlertDispatcher:
    def __init__(self, sender: Sender | None = None, site_name: str = "", max_queue: int = 500,
                 retry_base_s: float = 2.0, retry_max_s: float = 120.0, sleep=None,
                 min_confidence: float = 0.6, confident_only: bool = True, store=None,
                 tz: str | None = None, late_after_s: float = 120.0, max_age_s: float = 1800.0, clock=time.time):
        self.sender = sender
        self.tz = tz
        # Очередь при обрыве сети (S4): позже late_after_s — с пометкой «с опозданием»; старше max_age_s — не шлём
        # (через полчаса «гости ждут» уже бесполезно и только раздражает), но пишем в журнал.
        self.late_after_s = late_after_s
        self.max_age_s = max_age_s
        self.expired = 0
        self._clock = clock
        self.min_confidence = min_confidence
        self.confident_only = confident_only
        self.store = store  # EventStore: журнал алертов (alerts_sent) для отчёта и обратной связи
        self.held_back = 0
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
            conf = is_confident(e, self.min_confidence)
            wall = self._clock()
            rec = AlertRecord(e.ts, e.camera_id, e.zone_id, e.payload.get("kind", ""),
                              format_alert(e, self.site_name, (zone_titles or {}).get(e.zone_id), wall, self.tz),
                              confident=conf, alert_id=alert_id(e), wall=wall)
            self.feed.append(rec)
            out.append(rec)
            to_tg = conf or not self.confident_only
            log.warning("АЛЕРТ %s%s", rec.text, "" if to_tg else " (неуверенный — только в отчёт)")
            if not to_tg:
                self.held_back += 1
            self._log(rec)
            if self.sender is not None and to_tg:
                try:
                    self._q.put_nowait(rec)
                except queue.Full:
                    log.error("очередь алертов переполнена — алерт не отправлен: %s", rec.text)
        return out

    def _log(self, rec: AlertRecord) -> None:
        if self.store is None:
            return
        try:
            self.store.add_alert(rec.alert_id, rec.camera_id, rec.zone_id, rec.kind, rec.ts, rec.text,
                                 rec.confident, rec.sent, rec.wall)
        except Exception as e:  # noqa: BLE001
            log.debug("журнал алертов: %s", e)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rec = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            delay = self._retry_base
            while not self._stop.is_set():
                age = self._clock() - rec.wall
                if age > self.max_age_s:
                    self.expired += 1
                    log.warning("алерт устарел (%.0f мин без связи) — не отправлен: %s", age / 60, rec.text)
                    break
                text = rec.text if age <= self.late_after_s else f"{rec.text}\n{t('tg.late')}"
                try:
                    try:
                        self.sender.send(text, feedback_id=rec.alert_id)
                    except TypeError:  # отправитель без кнопки обратной связи
                        self.sender.send(text)
                    rec.sent = True
                    self.sent_count += 1
                    self._log(rec)
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
