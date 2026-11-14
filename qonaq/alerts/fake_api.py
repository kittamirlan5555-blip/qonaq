"""Локальный фейковый Telegram Bot API (S4): тот же формат запросов/ответов, что у api.telegram.org.

Нужен, чтобы проверить весь поток сообщений без интернета и без настоящего бота:
алерт администратору, кнопка «Неверно», сводка смены владельцу заведения, техническая сводка, очередь при обрыве.

    python -m qonaq telegram fake-server --port 8081
    TELEGRAM_API_BASE=http://127.0.0.1:8081 TELEGRAM_BOT_TOKEN=test TELEGRAM_CHAT_ID=100 python -m qonaq live ...

Методы: getMe, sendMessage, sendPhoto, sendMediaGroup, answerCallbackQuery, getUpdates (с long polling).
Служебные ручки (их у настоящего Telegram нет):
  GET  /__fake/messages            — всё, что «отправил» бот (по чатам)
  POST /__fake/press               — {"message_id": N} или {"alert_id": "..."}: нажать кнопку под сообщением
  POST /__fake/offline             — {"on": true}: имитировать обрыв (все методы отвечают 502)
  POST /__fake/reset               — очистить
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class FakeTelegramState:
    def __init__(self, token: str | None = None):
        self.token = token  # None — принимать любой
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "lock", threading.Lock()):
            self.messages: list[dict[str, Any]] = []
            self.updates: list[dict[str, Any]] = []
            self.answered: list[dict[str, Any]] = []
            self.offline = False
            self._mid = 0
            self._uid = 0
            self.calls: dict[str, int] = {}

    def add_message(self, method: str, chat_id: Any, text: str = "", reply_markup: Any = None,
                    photos: int = 0) -> dict[str, Any]:
        with self.lock:
            self._mid += 1
            m = {"message_id": self._mid, "method": method, "chat_id": str(chat_id), "text": text,
                 "reply_markup": reply_markup, "photos": photos, "date": int(time.time())}
            self.messages.append(m)
            return m

    def press(self, message_id: int | None = None, alert_id: str | None = None, user: str = "admin") -> dict:
        with self.lock:
            msg = None
            for m in reversed(self.messages):
                buttons = ((m.get("reply_markup") or {}).get("inline_keyboard") or [[]])[0]
                data = buttons[0].get("callback_data", "") if buttons else ""
                by_id = message_id is not None and m["message_id"] == message_id
                if by_id or (alert_id and data == f"fb:{alert_id}"):
                    msg = (m, data)
                    break
            if msg is None:
                raise KeyError("сообщение с кнопкой не найдено")
            self._uid += 1
            upd = {"update_id": self._uid, "callback_query": {
                "id": f"cq{self._uid}", "from": {"id": 7, "username": user}, "data": msg[1],
                "message": {"message_id": msg[0]["message_id"], "chat": {"id": msg[0]["chat_id"]}}}}
            self.updates.append(upd)
            return upd


def create_fake_app(state: FakeTelegramState | None = None):
    st = state or FakeTelegramState()
    app = FastAPI(title="Fake Telegram Bot API")
    app.state.fake = st

    def ok(result: Any) -> JSONResponse:
        return JSONResponse({"ok": True, "result": result})

    def err(code: int, desc: str) -> JSONResponse:
        return JSONResponse({"ok": False, "error_code": code, "description": desc}, status_code=code)

    async def body_of(req: Request) -> dict[str, Any]:
        ctype = req.headers.get("content-type", "")
        if "application/json" in ctype:
            return await req.json()
        raw = await req.body()
        out: dict[str, Any] = {"_files": 0}
        if "multipart/form-data" in ctype:  # без python-multipart: разбор стандартной библиотекой
            import email
            import email.policy

            msg = email.message_from_bytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw,
                                           policy=email.policy.HTTP)
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    out["_files"] += 1
                elif name:
                    out[name] = part.get_payload(decode=True).decode("utf-8", "replace")
            return out
        from urllib.parse import parse_qsl

        out.update(dict(parse_qsl(raw.decode("utf-8", "replace"))))
        return out

    @app.get("/__fake/messages")
    def fake_messages():
        with st.lock:
            return {"messages": list(st.messages), "answered": list(st.answered), "calls": dict(st.calls)}

    @app.post("/__fake/press")
    async def fake_press(req: Request):
        b = await req.json()
        try:
            return st.press(b.get("message_id"), b.get("alert_id"), b.get("user", "admin"))
        except KeyError as e:
            return err(404, str(e))

    @app.post("/__fake/offline")
    async def fake_offline(req: Request):
        st.offline = bool((await req.json()).get("on", True))
        return {"offline": st.offline}

    @app.post("/__fake/reset")
    def fake_reset():
        st.reset()
        return {"ok": True}

    @app.api_route("/bot{token}/{method}", methods=["GET", "POST"])
    async def bot_method(token: str, method: str, req: Request):
        st.calls[method] = st.calls.get(method, 0) + 1
        if st.offline:
            return err(502, "Bad Gateway (fake offline)")
        if st.token is not None and token != st.token:
            return err(401, "Unauthorized")
        b = await body_of(req) if req.method == "POST" else dict(req.query_params)
        if method == "getMe":
            return ok({"id": 1, "is_bot": True, "first_name": "Qonaq (fake)", "username": "qonaq_fake_bot"})
        if method == "sendMessage":
            if not b.get("chat_id") or not b.get("text"):
                return err(400, "Bad Request: chat_id and text are required")
            m = st.add_message(method, b["chat_id"], b["text"], b.get("reply_markup"))
            return ok({"message_id": m["message_id"], "chat": {"id": b["chat_id"]}, "text": b["text"]})
        if method == "sendPhoto":
            if not b.get("chat_id") or not b.get("_files"):
                return err(400, "Bad Request: there is no photo in the request")
            m = st.add_message(method, b["chat_id"], b.get("caption", ""), photos=1)
            return ok({"message_id": m["message_id"], "chat": {"id": b["chat_id"]}})
        if method == "sendMediaGroup":
            media = json.loads(b.get("media") or "[]")
            if not b.get("chat_id") or not media or b.get("_files", 0) < len(media):
                return err(400, "Bad Request: wrong media group")
            caps = " | ".join(x.get("caption", "") for x in media)
            m = st.add_message(method, b["chat_id"], caps, photos=len(media))
            return ok([{"message_id": m["message_id"], "chat": {"id": b["chat_id"]}}])
        if method == "answerCallbackQuery":
            with st.lock:
                st.answered.append({"callback_query_id": b.get("callback_query_id"), "text": b.get("text", "")})
            return ok(True)
        if method == "getUpdates":
            offset = int(b.get("offset") or 0)
            timeout = min(float(b.get("timeout") or 0), 30.0)
            deadline = time.time() + timeout
            while True:
                with st.lock:
                    if offset:
                        st.updates = [u for u in st.updates if u["update_id"] >= offset]  # подтверждённые — забыть
                    pending = list(st.updates)
                if pending or time.time() >= deadline:
                    return ok(pending)
                await asyncio.sleep(0.05)
        return err(404, f"Not Found: method {method}")

    return app


class FakeTelegramServer:
    """Фейковый API в фоновом потоке (uvicorn) — для `qonaq telegram test --fake` и ручной проверки."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8081, token: str | None = None):
        self.state = FakeTelegramState(token)
        self.host, self.port = host, port
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> FakeTelegramServer:
        import uvicorn

        cfg = uvicorn.Config(create_fake_app(self.state), host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, name="fake-telegram", daemon=True)
        self._thread.start()
        t0 = time.time()
        while not getattr(self._server, "started", False) and time.time() - t0 < 10:
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
