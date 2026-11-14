"""Исправления после первого запуска на Windows + RTX 2060 SUPER с настоящим Telegram (14.11.2026)."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace


def _fake_ort(available, preload=None):
    ns = SimpleNamespace(get_available_providers=lambda: list(available))
    if preload is not None:
        ns.preload_dlls = preload
    return ns


def test_cuda_provider_preloads_dlls_and_device_follows_session():
    from qonaq.detect.onnx_yolox import device_from_providers, select_providers

    called = []
    ort = _fake_ort(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                    preload=lambda: called.append(1))
    assert select_providers(ort, "auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert called, "DLL CUDA/cuDNN из pip-пакетов nvidia-* должны подгружаться до создания сессии"
    # onnxruntime-gpu без DLL CUDA: сессия молча работает на CPU — устройство должно говорить правду
    assert device_from_providers(["CPUExecutionProvider"]) == "cpu"
    assert device_from_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]) == "cuda:0"
    assert device_from_providers([]) == "cpu"
    # старый onnxruntime без preload_dlls и ошибка preload — не падаем
    assert select_providers(_fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"]), "cuda")[0] == \
        "CUDAExecutionProvider"

    def boom():
        raise RuntimeError("no nvidia packages")

    assert select_providers(_fake_ort(["CUDAExecutionProvider"], boom), "auto")[0] == "CUDAExecutionProvider"
    assert select_providers(_fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"]), "cpu") == \
        ["CPUExecutionProvider"]
    assert select_providers(_fake_ort(["DmlExecutionProvider", "CPUExecutionProvider"]), "auto")[0] == \
        "DmlExecutionProvider"


def test_long_polling_get_updates_does_not_hit_http_timeout():
    """FeedbackPoller ждёт getUpdates 20 с, а клиент httpx — 10 с: каждый опрос падал по таймауту
    (замечено на настоящем Telegram), и кнопка «Неверно» подхватывалась с задержкой до 2 минут."""
    from qonaq.alerts.telegram import TelegramSender

    seen = {}

    class Client:
        def post(self, url, json=None, **kw):
            seen[url.rsplit("/", 1)[-1]] = kw
            return SimpleNamespace(status_code=200, json=lambda: {"ok": True, "result": []})

    s = TelegramSender("T", "1", client=Client(), base_url="http://x")
    s.call("getUpdates", offset=0, timeout=20)
    s.call("answerCallbackQuery", callback_query_id="1")
    assert seen["getUpdates"]["timeout"] > 20
    assert "timeout" not in seen["answerCallbackQuery"]


def test_real_bot_test_sends_alert_with_button_and_shift_then_records_press(tmp_path):
    """`telegram test` на настоящем боте: проверка связи + пример алерта с кнопкой + пример итогов дня;
    `--wait-press` ловит нажатие «Неверно» и пишет его в feedback."""
    from fastapi.testclient import TestClient

    from qonaq.alerts.cli_telegram import EXAMPLE_ALERT_PREFIX, _site, send_examples
    from qonaq.alerts.fake_api import FakeTelegramState, create_fake_app
    from qonaq.alerts.telegram import TelegramSender
    from qonaq.events.store import EventStore

    state = FakeTelegramState(token="T0K")
    client = TestClient(create_fake_app(state))
    state.updates.append({"update_id": 1, "message": {"chat": {"id": 100}, "text": "привет"}})  # старое
    state._uid = 1

    def presser():
        t0 = time.time()
        while time.time() - t0 < 5:
            if any(m.get("reply_markup") for m in state.messages):
                state.press(message_id=next(m for m in state.messages if m.get("reply_markup"))["message_id"],
                            user="tester")
                return
            time.sleep(0.02)

    th = threading.Thread(target=presser, daemon=True)
    th.start()
    store = EventStore(tmp_path / "events.sqlite")
    r = send_examples(lambda chat: TelegramSender("T0K", chat, client=client, base_url="http://testserver"),
                      {"TELEGRAM_CHAT_ID": "100", "TELEGRAM_VENUE_CHAT_ID": "200"}, _site(None), store,
                      wait_press_s=5, poll_timeout_s=0, log=lambda *_: None)
    th.join(1)
    assert r["alert_sent"] and r["shift_sent"]
    admin = [m for m in state.messages if m["chat_id"] == "100"]
    assert admin[0]["text"].startswith(EXAMPLE_ALERT_PREFIX) and "Стол 5" in admin[0]["text"]
    assert admin[0]["reply_markup"]["inline_keyboard"][0][0]["text"] == "Неверно"
    assert any("итогов дня" in m["text"] for m in state.messages if m["chat_id"] == "200")
    assert len(r["feedback"]) == 1 and r["feedback"][0]["zone_id"] == "T05" and r["feedback"][0]["user"] == "tester"
    assert state.answered, "бот должен ответить на нажатие"
    store.close()


def test_shift_example_goes_to_admin_when_no_venue_chat(tmp_path):
    from fastapi.testclient import TestClient

    from qonaq.alerts.cli_telegram import _site, send_examples
    from qonaq.alerts.fake_api import FakeTelegramState, create_fake_app
    from qonaq.alerts.telegram import TelegramSender
    from qonaq.events.store import EventStore

    state = FakeTelegramState()
    client = TestClient(create_fake_app(state))
    store = EventStore(":memory:")
    r = send_examples(lambda chat: TelegramSender("x", chat, client=client, base_url="http://testserver"),
                      {"TELEGRAM_CHAT_ID": "100"}, _site(None), store, log=lambda *_: None)
    assert r["alert_sent"] and r["shift_sent"] and r["feedback"] == []
    assert {m["chat_id"] for m in state.messages} == {"100"} and len(state.messages) == 2
    store.close()


def test_demo_reel_without_datasets_explains_what_to_run(tmp_path, monkeypatch, capsys):
    """Раньше — FileNotFoundError data/index/roles.csv с трейсбеком; в PROGRESS про датасеты не было ни слова."""
    import argparse

    import qonaq.config
    from qonaq.cli import cmd_demo_reel

    scen = qonaq.config.PROJECT_ROOT / "configs" / "synth" / "demo_reel.yaml"
    monkeypatch.setattr(qonaq.config, "PROJECT_ROOT", tmp_path)
    args = argparse.Namespace(scenario=str(scen), rerun=False, site=None, out=str(tmp_path / "r.mp4"))
    assert cmd_demo_reel(args) == 2
    out = capsys.readouterr().out
    assert "fetch_datasets.py" in out and "data index" in out


def test_all_site_configs_load_and_reference_existing_zone_files():
    from qonaq.config import PROJECT_ROOT, load_site

    for p in sorted((PROJECT_ROOT / "configs" / "sites").glob("*.yaml")):
        site = load_site(p)
        for cam in site.cameras:
            if cam.zones:
                assert (PROJECT_ROOT / cam.zones).exists(), f"{p.name}: нет {cam.zones}"
