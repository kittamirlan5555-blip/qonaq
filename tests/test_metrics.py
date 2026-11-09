"""M6: метрики и ROI считаются из событий; числа отчёта сходятся с журналом."""
import json

import pytest

from qonaq.config import AlertThresholds, RestaurantThresholds, SiteConfig
from qonaq.events.store import AsyncEventWriter, EventStore
from qonaq.logic.base import Event
from qonaq.logic.restaurant import TableMachine
from qonaq.metrics.compute import compute_metrics, merge_intervals, overlap_len, percentile, stats
from qonaq.metrics.report import dur, generate_report, html_report, markdown_report, telegram_text
from qonaq.strings import fmt_money, guests_word
from tests.helpers import Timeline

G, W = (1, 2), (50,)


def run_table(store: EventStore, zone: str, script, cam: str = "cam") -> Timeline:
    m = TableMachine(zone, RestaurantThresholds(preexisting_grace_s=5), AlertThresholds())
    tl = Timeline(m, zone_id=zone)
    for seconds, kw in script:
        tl.run(seconds, **kw)
    for e in tl.events:
        e.camera_id = cam
    store.add(tl.events)
    return tl


def shift_store() -> tuple[EventStore, dict]:
    """Синтетическая «смена» 1000 с на 2 столах с известными ответами."""
    store = EventStore()
    # T01: сели в 20, подошли в 80 (ожидание 60), ушли в 300, убрали в 500..515 (грязный ~215 с)
    t1 = run_table(store, "T01", [
        (20, {}), (60, {"guests": G}), (8, {"guests": G, "staff": W}), (212, {"guests": G}),
        (200, {}), (15, {"staff": W}), (485, {}),
    ])
    # T02: сели в 30, подошли в 270 (ожидание 240 > 180 -> алерт), сидят до конца
    t2 = run_table(store, "T02", [
        (30, {}), (240, {"guests": G}), (8, {"guests": G, "staff": W}), (722, {"guests": G}),
    ])
    store.set_meta(time_base="video", scene_s=1000.0, tables=2)
    return store, {"t1": t1, "t2": t2}


def test_helpers():
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([10], 90) == 10 and percentile([], 50) is None
    assert stats([1, 2, 3])["mean"] == 2 and stats([])["n"] == 0
    assert merge_intervals([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]
    assert overlap_len((4, 11), [(0, 8), (10, 12)]) == 5
    assert dur(125) == "2 мин 05 с" and dur(None) == "—" and dur(3700) == "1 ч 01 мин"
    assert fmt_money(45000) == "45 000 ₸"
    assert [guests_word(n) for n in (1, 2, 5, 11, 21)] == ["1 гость", "2 гостя", "5 гостей", "11 гостей", "21 гость"]


def test_sessions_are_built_from_events():
    store, _ = shift_store()
    ss = {s["zone_id"]: s for s in store.sessions()}
    assert ss["T01"]["outcome"] == "completed" and ss["T02"]["outcome"] == "open"
    assert ss["T01"]["response_time_s"] == pytest.approx(60, abs=1.5)
    assert ss["T01"]["visit_duration_s"] == pytest.approx(280, abs=2)
    assert ss["T01"]["cleanup_time_s"] == pytest.approx(210, abs=3)
    assert ss["T02"]["response_time_s"] == pytest.approx(240, abs=1.5)


def test_metrics_match_events():
    store, tl = shift_store()
    m = compute_metrics(store, SiteConfig())
    served = tl["t1"].of("served") + tl["t2"].of("served")
    rts = sorted(e.payload["response_time_s"] for e in served)
    assert m["response"]["n"] == 2
    assert m["response"]["mean"] == pytest.approx(sum(rts) / 2, abs=0.1)
    assert m["response"]["median"] == pytest.approx(sum(rts) / 2, abs=0.1)
    assert m["response"]["share_over_threshold"] == 0.5  # 240 с > 180 с
    assert m["alerts"] == {"no_waiter": 1}
    assert m["dirty"]["n"] == 1
    assert m["dirty"]["mean"] == pytest.approx(tl["t1"].of("cleaned")[0].payload["cleanup_time_s"], abs=0.1)
    assert m["turnover"]["seatings"] == 2 and m["turnover"]["per_table"] == 1.0
    assert m["tables"]["total"] == 2 and m["period"]["duration_s"] == pytest.approx(1000, abs=1)
    # загрузка: T01 занят ~280 с, T02 ~970 с из 1000
    assert m["tables"]["avg_load"] == pytest.approx((280 + 970) / 2000, abs=0.01)
    assert m["problem_tables"][0]["zone_id"] == "T02"


def test_roi_counts_only_peak_dirty_time():
    """Грязный стол стоит денег только в пик. Пик: загрузка >= 80% => для 2 столов нужны оба занятых."""
    store, _ = shift_store()
    site = SiteConfig()
    m = compute_metrics(store, site)
    # Оба стола заняты только 30..300 с; T01 грязный 300..~510 — уже вне пика => потерь нет.
    assert m["roi"]["peak_s"] == pytest.approx(270, abs=3)
    assert m["roi"]["excess_dirty_in_peak_s"] == 0 and m["roi"]["loss_kzt"] == 0
    # Если «пиком» считать загрузку >= 50%, пик = всё время, пока занят хотя бы один стол.
    site.business.peak_occupancy = 0.5
    site.business.target_cleanup_s = 120
    site.business.default_visit_min = 60
    site.business.avg_check_kzt = 15000
    m2 = compute_metrics(store, site)
    dirty = m2["dirty"]["mean"]
    assert m2["roi"]["excess_dirty_in_peak_s"] == pytest.approx(dirty - 120, abs=1)
    lost = (dirty - 120) / 3600
    assert m2["roi"]["lost_seatings"] == pytest.approx(lost, abs=0.01)
    assert m2["roi"]["loss_kzt"] == pytest.approx(lost * 15000, abs=60)


def test_entrance_queue_makes_peak():
    store, _ = shift_store()
    store.add([Event(350.0, "E01", "queue", {"count": 2}, "cam"), Event(450.0, "E01", "queue", {"count": 0}, "cam")])
    m = compute_metrics(store, SiteConfig())
    assert m["queue"]["max"] == 2 and m["queue"]["time_with_queue_s"] == pytest.approx(100)
    assert m["roi"]["peak_s"] == pytest.approx(370, abs=3)  # 270 по загрузке + 100 по очереди
    assert m["roi"]["excess_dirty_in_peak_s"] == 0  # 100 с грязного стола в пик < норматива 120 с


def test_walkout_and_dirty_seat_metrics():
    store = EventStore()
    run_table(store, "T01", [(20, {}), (100, {"guests": G}), (100, {}), (300, {})])  # ушли не дождавшись
    run_table(store, "T02", [(20, {}), (20, {"guests": G}), (6, {"guests": G, "staff": W}), (60, {"guests": G}),
                             (120, {}), (60, {"guests": (7, 8)}), (194, {"guests": (7, 8)})])
    store.set_meta(time_base="video", scene_s=520.0, tables=2)
    m = compute_metrics(store, SiteConfig())
    assert m["walkouts"]["n"] == 1 and m["roi"]["walkout_loss_kzt"] == 15000
    assert m["dirty"]["seated_at_dirty"] == 1
    assert m["dirty"]["n"] == 1 and m["dirty"]["mean"] == pytest.approx(120, abs=2)
    assert m["still_waiting"] == 1


def test_preexisting_sessions_excluded_from_response_time():
    store = EventStore()
    run_table(store, "T01", [(60, {"guests": G}), (6, {"guests": G, "staff": W}), (60, {"guests": G})])
    store.set_meta(time_base="video", scene_s=126.0, tables=1)
    m = compute_metrics(store, SiteConfig())
    assert m["response"]["n"] == 0 and m["quality"]["preexisting_sessions"] == 1


def test_report_files_and_numbers(tmp_path):
    store, _ = shift_store()
    store.close()
    # тот же журнал, но на диске
    disk = EventStore(tmp_path / "events.sqlite")
    src, tl = shift_store()
    for t in tl.values():
        disk.add(t.events)
    disk.set_meta(time_base="video", scene_s=1000.0, tables=2)
    disk.close()
    (tmp_path / "run.json").write_text(json.dumps({"site_id": "demo", "name": "Тест-кафе", "profile": "restaurant",
                                                    "zones": {"cam": ["T01", "T02", "E01"]}}), encoding="utf-8")
    out = generate_report(tmp_path)
    m = json.loads(out["json"].read_text(encoding="utf-8"))
    html_text = out["html"].read_text(encoding="utf-8")
    md = out["md"].read_text(encoding="utf-8")
    tg = out["telegram"].read_text(encoding="utf-8")
    assert "Тест-кафе" in html_text and "<html" in html_text
    assert dur(m["response"]["median"]) in html_text and dur(m["response"]["median"]) in md
    assert dur(m["response"]["median"]) in tg
    assert len(tg.splitlines()) <= 10
    assert "Оговорки о точности" in md and "верить" in html_text
    assert "₸" in html_text


def test_report_renders_with_no_data():
    store = EventStore()
    store.set_meta(time_base="video", scene_s=60.0, tables=3)
    m = compute_metrics(store, SiteConfig())
    assert m["response"]["n"] == 0 and m["tables"]["total"] == 3
    assert "нет измерений" in telegram_text(m)
    assert html_report(m) and markdown_report(m)


def test_async_writer_does_not_lose_events(tmp_path):
    store = EventStore(tmp_path / "e.sqlite")
    seen = []
    w = AsyncEventWriter(store, on_events=seen.extend)
    for i in range(50):
        w.add([Event(float(i), "T01", "queue", {"count": i}, "cam")])
    w.close()
    assert len(store.events()) == 50 and len(seen) == 50
    assert store.events(limit=3, newest_first=True)[0]["payload"]["count"] == 49
