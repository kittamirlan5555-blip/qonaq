"""HTML-алерты Telegram: карточка, кнопки «Принял»/«Неверно», правка алерта при подходе/уборке, итог смены."""
from __future__ import annotations

import time

from qonaq.alerts.feed import TelegramFeed, alert_card, dur, keyboard, shift_html, shift_report
from qonaq.logic.base import Event


def clock(ts: float) -> str:
    """Время журнала (секунды от 19:00) -> ЧЧ:ММ."""
    m = int(19 * 60 + ts // 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def ev(ts, zone, typ, **p):
    return Event(float(ts), zone, typ, p, "cam")


def test_durations():
    assert [dur(x) for x in (40, 220, 180, 540, 3900)] == ["40 с", "3 мин 40 с", "3 мин", "9 мин", "1 ч 05 мин"]
    assert dur(76.5) == "1 мин 17 с"  # половина секунды — вверх, как на карточках


def test_alert_cards_are_strict_html():
    assert alert_card("no_waiter", "Стол 4", 180, "19:42") == (
        "<b>Стол 4 · гости ждут официанта 3 мин</b>\nСели в 19:42, к столу пока никто не подходил.")
    assert alert_card("dirty_table", "Стол 8", 360, "20:05") == (
        "<b>Стол 8 · не убран 6 мин</b>\nГости ушли в 20:05, после этого к столу не подходили.")
    assert alert_card("no_waiter", "Стол <1>", 180, None).startswith("<b>Стол &lt;1&gt;")  # экранирование
    kb = keyboard("a1")["inline_keyboard"][0]
    assert [b["text"] for b in kb] == ["Принял", "Неверно"] and kb[1]["callback_data"] == "fb:a1"


def test_feed_sends_alert_then_edits_when_waiter_came():
    f = TelegramFeed(clock, titles={"T04": "Стол 4"})
    acts = f.handle([ev(42 * 60, "T04", "seated", sid=1), ev(45 * 60, "T04", "alert", kind="no_waiter", sid=1,
                                                              waited_s=180.0)])
    assert [a.op for a in acts] == ["send"]
    assert acts[0].html == "<b>Стол 4 · гости ждут официанта 3 мин</b>\nСели в 19:42, к столу пока никто не подходил."
    edit = f.handle([ev(45 * 60 + 40, "T04", "served", sid=1, response_time_s=220.0)])
    assert len(edit) == 1 and edit[0].op == "edit" and edit[0].key == acts[0].key
    assert edit[0].html.endswith("\n<i>Подошли через 3 мин 40 с</i>")
    assert [b["text"] for b in edit[0].markup["inline_keyboard"][0]] == ["Неверно"]
    assert f.handle([ev(46 * 60, "T04", "served", sid=1, response_time_s=1.0)]) == []  # второй раз не правим


def test_feed_dirty_table_closed_by_cleaning_of_same_session():
    f = TelegramFeed(clock)
    f.handle([ev(65 * 60, "T08", "left", sid=3), ev(71 * 60, "T08", "alert", kind="dirty_table", sid=3,
                                                    dirty_s=360.0)])
    assert f.handle([ev(74 * 60, "T08", "cleaned", sid=2, cleanup_time_s=10.0)]) == []  # чужая сессия
    e = f.handle([ev(74 * 60, "T08", "cleaned", sid=3, cleanup_time_s=540.0)])
    assert "Гости ушли в 20:05" in e[0].html and e[0].html.endswith("<i>Убрали через 9 мин</i>")


def _sessions():
    return [  # время — секунды от 19:00
        {"zone_id": "T01", "seated_ts": 0, "served_ts": 100, "response_time_s": 100, "left_ts": 2400,
         "visit_duration_s": 2400, "cleaned_ts": 2640, "cleanup_time_s": 240, "preexisting": 0},
        {"zone_id": "T03", "seated_ts": 2400, "served_ts": 2780, "response_time_s": 380, "left_ts": 5400,
         "visit_duration_s": 3000, "cleaned_ts": 5800, "cleanup_time_s": 400, "preexisting": 0},
        {"zone_id": "T02", "seated_ts": -600, "served_ts": None, "response_time_s": None, "left_ts": 900,
         "visit_duration_s": None, "cleaned_ts": 960, "cleanup_time_s": 60, "preexisting": 1},
    ]


def test_shift_report_counts_only_from_journal():
    evs = [{"ts": 2580.0, "zone_id": "T03", "type": "alert"}, {"ts": 5700.0, "zone_id": "T03", "type": "alert"},
           {"ts": 100.0, "zone_id": "T01", "type": "alert"}]
    r = shift_report(_sessions(), evs, clock, hour_of=lambda t: 19 + int(t // 3600), t_end=6000)
    assert r["seatings"] == 2  # preexisting не считается
    assert r["stay_mean_s"] == 2700 and r["resp_mean_s"] == 240 and r["long_wait_n"] == 1
    assert r["resp_worst"] == {"s": 380, "table": "Стол 3", "at": "19:40"}
    assert r["dirty_over_n"] == 1 and r["dirty_total_s"] == 700 and r["problem_table"] == "Стол 3"
    assert r["peak_hour"] in (19, 20)
    html = shift_html(r, "Зал · демо", "18 ноября")
    assert html.startswith("<b>Итоги смены</b>\nЗал · демо · 18 ноября\n\n<b>Гости</b>\nПосадок за столы — 2")
    assert "Дольше всего ждали — 6 мин 20 с, стол 3 в 19:40" in html and "Ждали дольше 3 мин — 1 раз" in html
    assert "Стояли неубранными дольше 5 мин — 1 раз" in html and "Чаще всего проблемы — стол 3" in html


def test_shift_html_drops_lines_without_measurements():
    r = shift_report([{"zone_id": "T01", "seated_ts": 0, "left_ts": None, "preexisting": 0}], [], clock,
                     hour_of=lambda t: 19, t_end=60)
    html = shift_html(r, "Зал", "1 мая")
    assert "Подход" not in html and "Уборка" not in html and "Неверно" not in html and "не измерен" not in html
    assert "<b>Обслуживание</b>" not in html and "Посадок за столы — 1" in html


def test_dispatcher_html_on_fake_api_send_edit_ack():
    from fastapi.testclient import TestClient

    from qonaq.alerts.fake_api import FakeTelegramState, create_fake_app
    from qonaq.alerts.telegram import AlertDispatcher, FeedbackPoller, TelegramSender
    from qonaq.events.store import EventStore

    state = FakeTelegramState(token="T0K")
    client = TestClient(create_fake_app(state))
    sender = TelegramSender("T0K", "100", client=client, base_url="http://testserver")
    d = AlertDispatcher(sender, html=True, confident_only=False, tz="Asia/Almaty")
    d.handle([ev(1000, "T04", "seated", sid=1),
              ev(1180, "T04", "alert", kind="no_waiter", sid=1, waited_s=180.0, by_role=True, confidence=0.9)],
             {"T04": "Стол 4"})

    def wait(cond):
        t0 = time.time()
        while not cond() and time.time() - t0 < 5:
            time.sleep(0.02)
        return cond()

    assert wait(lambda: len(state.messages) == 1)
    m = state.messages[0]
    assert m["text"].startswith("<b>Стол 4 · гости ждут официанта 3 мин</b>")
    store = EventStore(":memory:")
    poller = FeedbackPoller(sender, store, timeout_s=0, on_ack=d.ack)
    state.press(message_id=m["message_id"], button="ack")
    poller.poll_once()
    assert wait(lambda: "<i>Принял" in state.messages[0]["text"])
    d.handle([ev(1400, "T04", "served", sid=1, response_time_s=400.0)])
    assert wait(lambda: state.messages[0]["text"].endswith("<i>Подошли через 6 мин 40 с</i>"))
    assert len(state.messages) == 1 and state.messages[0].get("edits") == 2
    assert [f["verdict"] for f in store.feedback()] == ["ack"]
    d.close()


def test_held_back_alert_is_not_sent_in_html_mode():
    class S:
        def __init__(self):
            self.sent = []

        def send_html(self, html, markup=None):
            self.sent.append(html)
            return len(self.sent)

        def edit_html(self, *a):
            pass

        def send(self, text):
            pass

    from qonaq.alerts.telegram import AlertDispatcher

    s = S()
    d = AlertDispatcher(s, html=True, confident_only=True)
    d.handle([ev(10, "T01", "alert", kind="no_waiter", sid=1, waited_s=180.0, by_role=False, confidence=0.3)])
    time.sleep(0.3)
    assert s.sent == [] and d.held_back == 1
    d.close()


def test_session_cut_by_segment_end_is_not_counted_as_long_wait():
    """Показ из отрезков: сели за 90 с до конца отрезка, подход за окном — это не «ждали дольше 3 мин»."""
    s = [{"zone_id": "T02", "seated_ts": 0, "served_ts": None, "left_ts": None, "preexisting": 0, "t_end": 90}]
    r = shift_report(s, [], clock, hour_of=lambda t: 19, t_end=5000)
    assert r["long_wait_n"] == 0 and r["seatings"] == 1
