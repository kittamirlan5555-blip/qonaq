"""`python -m qonaq telegram ...` (S4): test | chat-id | fake-server | examples."""
from __future__ import annotations

import os
import time
from typing import Any

from qonaq.logic.base import Event

CHATS = [
    ("TELEGRAM_CHAT_ID", "администратор зала (алерты)"),
    ("TELEGRAM_VENUE_CHAT_ID", "владелец заведения (итоги дня)"),
    ("TELEGRAM_OWNER_CHAT_ID", "владелец системы (техническая сводка)"),
]


def add_parser(sub) -> None:
    tp = sub.add_parser("telegram", help="Telegram: test | chat-id | fake-server | examples (S4)")
    t = tp.add_subparsers(dest="tg_cmd", required=True)
    p = t.add_parser("test", help="тестовое сообщение во все настроенные чаты (.env)")
    p.add_argument("--fake", action="store_true", help="поднять локальный фейковый API и проверить весь поток на нём")
    p.add_argument("--site", help="YAML объекта (название, часовой пояс)")
    p = t.add_parser("chat-id", help="узнать chat id: напишите боту (или добавьте его в группу) и запустите")
    p = t.add_parser("fake-server", help="локальный фейковый Telegram Bot API для проверки")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p = t.add_parser("examples", help="напечатать примеры всех сообщений (для docs/TELEGRAM.md)")
    p.add_argument("--site", help="YAML объекта")


def _site(path: str | None):
    from qonaq.config import SiteConfig, load_site

    return load_site(path) if path else SiteConfig(name="Кафе «Пример»")


def example_messages(site=None, wall: float | None = None) -> dict[str, Any]:
    """Все сообщения системы на синтетических данных — тем же кодом, что и в работе."""
    from qonaq.alerts.telegram import format_alert
    from qonaq.events.store import EventStore
    from qonaq.health.daily import summary_text
    from qonaq.metrics.compute import compute_metrics
    from qonaq.metrics.report import shift_summary_text

    site = site or _site(None)
    tz = site.business.timezone
    wall = wall or time.time()
    al = {
        "no_waiter": Event(0, "T05", "alert", {"kind": "no_waiter", "sid": 1, "waited_s": 192.0}),
        "dirty_table": Event(0, "T02", "alert", {"kind": "dirty_table", "sid": 2, "dirty_s": 315.0}),
        "long_service_without_visit": Event(0, "T07", "alert", {"kind": "long_service_without_visit", "sid": 3,
                                                                 "since_visit_s": 960.0}),
        "occupied_without_check": Event(0, "T03", "alert", {"kind": "occupied_without_check", "sid": 4,
                                                             "occupied_s": 640.0}),
    }
    alerts = {k: format_alert(e, "", None, wall, tz) for k, e in al.items()}
    # день на синтетике: 14 посадок, один уход не дождавшись, пиковое время с грязным столом
    store = EventStore(":memory:")
    t0 = wall - 10 * 3600
    evs: list[Event] = []
    for i in range(14):
        z = f"T0{1 + i % 6}"
        ts = t0 + i * 2400
        resp = [70, 95, 110, 130, 150, 85, 240, 300, 60, 100, 200, 90, 120, 75][i]
        evs += [Event(ts, z, "seated", {"sid": i, "guests": 2}, "cam1")]
        if i == 6:
            evs += [Event(ts + 420, z, "walkout_before_service", {"sid": i, "waited_s": 420}, "cam1")]
            continue
        evs += [Event(ts + resp, z, "served", {"sid": i, "response_time_s": resp}, "cam1"),
                Event(ts + 2000, z, "left", {"sid": i, "visit_duration_s": 2000 - resp}, "cam1"),
                Event(ts + 2000 + (600 if i == 9 else 90), z, "cleaned", {"sid": i, "cleanup_time_s": 90}, "cam1")]
    for e in evs:
        if e.type == "seated" and e.payload["sid"] in (7, 10):
            evs.append(Event(e.ts + 180, e.zone_id, "alert", {"kind": "no_waiter", "sid": e.payload["sid"],
                                                               "waited_s": 180}, "cam1"))
    store.add(sorted(evs, key=lambda e: e.ts))
    store.set_meta(time_base="epoch", tables=6, t_start=t0, t_end=wall)
    m = compute_metrics(store, site)
    from datetime import datetime

    try:
        from zoneinfo import ZoneInfo

        lt = datetime.fromtimestamp(wall, ZoneInfo(tz))
    except Exception:  # noqa: BLE001
        lt = datetime.fromtimestamp(wall)
    months = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря"]
    shift = shift_summary_text(m, site.business.currency, f"{lt.day} {months[lt.month - 1]}")
    status = {"status": "warn", "issues": [{"camera_id": "cam2", "message": "кадр тёмный 75 с (свет выключен или "
                                            "камера закрыта)"}],
              "cameras": {"cam1": {"outages": 1, "down_s_total": 130.0}, "cam2": {"outages": 0}}}
    perf = {"cpu_pct": 86.0, "rss_mb": 412.0, "cameras": {"cam1": {"fps": 2.0}, "cam2": {"fps": 1.9}}}
    daily = summary_text(site.name, status, store.events(), [{"confident": True}, {"confident": False}],
                         [{"alert_id": "x"}], perf)
    health = {
        "camera_down": f"🛠 {site.name} · cam2: камера не отвечает 45 с",
        "camera_recovered": f"🛠 {site.name} · cam2: камера снова работает (не было 130 с)",
    }
    store.close()
    return {"alerts": alerts, "shift": shift, "daily": daily, "health": health,
            "feedback_answer": "Записано: неверно. Спасибо!",
            "late_suffix": "Доставлено с опозданием: не было интернета."}


def cmd_examples(args) -> int:
    ex = example_messages(_site(args.site))
    print("== Алерты администратору (с кнопкой «Неверно») ==")
    for v in ex["alerts"].values():
        print(v + "\n")
    print("== При обрыве интернета (доставлено позже 2 мин) ==")
    print(next(iter(ex["alerts"].values())) + "\n" + ex["late_suffix"] + "\n")
    print("== Итоги дня владельцу заведения ==\n" + ex["shift"] + "\n")
    print("== Техническая сводка владельцу системы ==\n" + ex["daily"] + "\n")
    print("== Проблемы с камерами (владельцу системы) ==\n" + "\n".join(ex["health"].values()))
    return 0


def run_flow(sender_factory, fake_state=None, site=None) -> list[str]:
    """Полный поток на фейковом API: алерт с кнопкой → нажатие → feedback; итоги дня; тех. сводка с кадром;
    обрыв сети → очередь → доставка с пометкой; дедупликация. Возвращает список проверок."""
    import cv2
    import numpy as np

    from qonaq.alerts.telegram import AlertDispatcher, FeedbackPoller
    from qonaq.events.store import EventStore

    site = site or _site(None)
    ex = example_messages(site)
    checks: list[str] = []
    store = EventStore(":memory:")
    admin = sender_factory("100")
    clock = {"t": time.time()}
    disp = AlertDispatcher(admin, store=store, tz=site.business.timezone, retry_base_s=0.05, retry_max_s=0.1,
                           clock=lambda: clock["t"])
    e = Event(100.0, "T05", "alert", {"kind": "no_waiter", "sid": 1, "waited_s": 190.0}, "cam1")
    disp.handle([e, e])  # второй раз то же событие — дедупликация
    _wait(lambda: disp.sent_count >= 1)
    checks.append(f"алерт администратору отправлен: {disp.sent_count == 1}")
    if fake_state is not None:
        msgs = [m for m in fake_state.messages if m["chat_id"] == "100"]
        checks.append(f"дедупликация (одно сообщение на событие): {len(msgs) == 1}")
        fake_state.press(message_id=msgs[-1]["message_id"])
        n = FeedbackPoller(admin, store, timeout_s=0).poll_once()
        fb = store.feedback()
        checks.append(f"кнопка «Неверно» → feedback: {n == 1 and len(fb) == 1 and fb[0]['zone_id'] == 'T05'}")
        checks.append(f"ответ на нажатие: {bool(fake_state.answered)}")
        # обрыв: сеть пропала, алерт ждёт в очереди; связь вернулась через 3 «минуты» — уходит с пометкой
        fake_state.offline = True
        e2 = Event(300.0, "T02", "alert", {"kind": "dirty_table", "sid": 2, "dirty_s": 310.0}, "cam1")
        disp.handle([e2])
        _wait(lambda: disp.failed_attempts >= 2)
        clock["t"] += 180
        fake_state.offline = False
        _wait(lambda: disp.sent_count >= 2)
        late = [m for m in fake_state.messages if "опозданием" in m["text"]]
        checks.append(f"очередь при обрыве: доставлено после восстановления с пометкой: {len(late) == 1}")
    disp.close()
    venue = sender_factory("200")
    venue.send(ex["shift"])
    checks.append("итоги дня владельцу заведения: отправлено")
    owner = sender_factory("300")
    owner.send(ex["daily"])
    img = np.full((180, 320, 3), 90, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    owner.send_photos([(buf.tobytes(), "cam1 · 21:40 · T01: свободен")] * 2)
    owner.send_photo(buf.tobytes(), "cam1 · кадр")
    checks.append("техническая сводка + кадры владельцу системы: отправлено")
    store.close()
    return checks


def _wait(cond, timeout: float = 5.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def cmd_test(args) -> int:
    from qonaq.alerts.telegram import TelegramSender, api_base

    site = _site(args.site)
    if args.fake:
        import socket

        from qonaq.alerts.fake_api import FakeTelegramServer

        with socket.socket() as so:  # свободный порт
            so.bind(("127.0.0.1", 0))
            port = so.getsockname()[1]
        srv = FakeTelegramServer(port=port).start()
        try:
            checks = run_flow(lambda chat: TelegramSender("test", chat, base_url=srv.base_url), srv.state, site)
            for c in checks:
                print("•", c)
            print("\nЧто «получили» чаты:")
            for m in srv.state.messages:
                photos = f" [+{m['photos']} фото]" if m["photos"] else ""
                print(f"--- чат {m['chat_id']} ({m['method']}){photos}\n{m['text']}")
            bad = [c for c in checks if c.endswith("False")]
            return 1 if bad else 0
        finally:
            srv.stop()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Нет TELEGRAM_BOT_TOKEN в .env — см. docs/TELEGRAM.md (BotFather → токен). "
              "Проверить поток без бота: python -m qonaq telegram test --fake")
        return 2
    print(f"Bot API: {api_base()}")
    any_ok, failed = False, False
    for env, who in CHATS:
        chat = os.environ.get(env)
        if not chat:
            print(f"  {env} не задан — {who}: пропущено")
            continue
        try:
            TelegramSender(token, chat).send(f"✅ Qonaq · {site.name}: проверка связи. Сюда будут приходить: {who}.")
            print(f"  {env}={chat} — {who}: отправлено")
            any_ok = True
        except Exception as e:  # noqa: BLE001
            print(f"  {env}={chat} — {who}: ОШИБКА {e}")
            failed = True
    return 1 if failed or not any_ok else 0


def cmd_chat_id(args) -> int:
    from qonaq.alerts.telegram import TelegramSender

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Нет TELEGRAM_BOT_TOKEN в .env")
        return 2
    res = TelegramSender(token, "0").call("getUpdates", timeout=0)
    seen = {}
    for u in res.get("result", []):
        msg = u.get("message") or u.get("my_chat_member") or (u.get("callback_query") or {}).get("message") or {}
        chat = msg.get("chat") or {}
        if chat:
            seen[chat["id"]] = chat.get("title") or chat.get("username") or chat.get("first_name") or ""
    if not seen:
        print("Сообщений нет. Напишите боту /start (или добавьте бота в группу и напишите там), затем повторите.")
        return 1
    for cid, name in seen.items():
        print(f"chat id {cid}  —  {name}")
    return 0


def cmd_fake_server(args) -> int:
    from qonaq.alerts.fake_api import FakeTelegramServer

    srv = FakeTelegramServer(args.host, args.port).start()
    print(f"Фейковый Telegram Bot API: {srv.base_url}\n"
          f"  в .env: TELEGRAM_API_BASE={srv.base_url}  (TELEGRAM_BOT_TOKEN — любой)\n"
          f"  что отправлено: {srv.base_url}/__fake/messages   Ctrl+C — остановить")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


def run(args) -> int:
    return {"test": cmd_test, "chat-id": cmd_chat_id, "fake-server": cmd_fake_server,
            "examples": cmd_examples}[args.tg_cmd](args)
