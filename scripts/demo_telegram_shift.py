"""Показ Telegram-алертов и итога смены на записях: события берутся из журналов прогонов, тексты — шаблоны продукта.

    python scripts/demo_telegram_shift.py data/demo/shift.yaml              # только показать, что уйдёт в чат
    python scripts/demo_telegram_shift.py data/demo/shift.yaml --send       # отправить (TELEGRAM_BOT_TOKEN/CHAT_ID)

План (YAML, лежит в data/ — не коммитится):

    site: "Зал · демо"
    day: "16 ноября"
    segments:
      - name: hogs
        events: runs/full/hogs/events.sqlite
        clock0: "18:59:45"          # время сцены (OSD трансляции) в 0:00 исходного файла записи
        offset: 0                   # журнал прогона вырезки: 0:00 вырезки = offset секунд исходного файла
        window: [1080, 1700]        # секунды исходного файла: берём события внутри
        tables: {T01: "Стол 1", T04: "Стол 4"}   # только эти столы; номера в плане не пересекаются
        gt: tests/fixtures/gt/hogs_outside_1116.yaml   # сверка: время разметки — секунды файла

Ничего не дописывается руками: алерты, подтверждения («Подошли через …», «Убрали через …») и цифры итога —
`qonaq.alerts.feed` из событий и сессий журнала. Перед отправкой печатается каждое сообщение с исходным временем
файла и ближайшим событием разметки — для проверки глазами. Время в сообщениях — время сцены на записи.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qonaq.alerts.feed import TelegramFeed, shift_html, shift_report  # noqa: E402
from qonaq.config import load_dotenv_if_any  # noqa: E402
from qonaq.events.store import EventStore  # noqa: E402

DAY_S = 86400.0


def hms(s: str) -> float:
    h, m, *sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + (sec[0] if sec else 0)


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def load_segment(seg: dict, idx: int) -> tuple[list[dict], list[dict]]:
    """События и сессии отрезка → общая шкала «время сцены, секунды суток»; столы — под именами плана."""
    st = EventStore(ROOT / seg["events"])
    try:
        evs, sess = st.events(), st.sessions("table")
    finally:
        st.close()
    off = float(seg.get("offset", 0))
    a, b = (x - off for x in seg["window"])  # окно — в секундах исходного файла, журнал — от начала вырезки
    c0 = hms(seg["clock0"]) + off
    tables = seg["tables"]
    pref = f"s{idx}:"
    out_e = []
    for e in evs:
        if e["zone_id"] not in tables or not a <= e["ts"] <= b or e["type"] == "heartbeat":
            continue
        p = dict(e.get("payload") or {})
        e2 = {**e, "ts": c0 + e["ts"], "zone_id": pref + e["zone_id"], "payload": p, "camera_id": f"seg{idx}",
              "src": {"segment": seg["name"], "file_ts": e["ts"] + off, "zone": e["zone_id"]}}
        out_e.append(e2)
    out_s = []
    for s in sess:
        stamps = [s.get(k) for k in ("seated_ts", "served_ts", "left_ts", "cleaned_ts") if s.get(k) is not None]
        if s["zone_id"] not in tables or not any(a <= x <= b for x in stamps):
            continue
        s2 = dict(s)
        if s2["seated_ts"] < a:  # сели до окна отрезка: начало сессии в показе «не видно»
            s2["preexisting"] = 1
            s2["seated_ts"], s2["served_ts"], s2["response_time_s"], s2["visit_duration_s"] = a, None, None, None
        for k in ("seated_ts", "served_ts", "left_ts", "cleaned_ts"):
            if s2.get(k) is not None:
                if s2[k] > b:  # за окном отрезка — система этого «не видела» в показе
                    s2[k] = None
                    s2[{"served_ts": "response_time_s", "left_ts": "visit_duration_s",
                        "cleaned_ts": "cleanup_time_s"}.get(k, "x")] = None
                else:
                    s2[k] = c0 + s2[k]
        s2["zone_id"] = pref + s["zone_id"]
        s2["t_end"] = c0 + b  # дальше окна отрезка о столе ничего не утверждаем
        out_s.append(s2)
    return out_e, out_s


def gt_near(gt: dict | None, zone: str, typ: str, t: float, tol: float = 90.0) -> str:
    if not gt:
        return "нет разметки — проверить по кадру"
    alias = {"alert": None}
    best = None
    for g in gt.get("events", []):
        if g["zone"] != zone:
            continue
        if alias.get(typ, typ) is not None and g["event"] != typ:
            continue
        d = t - float(g["t"])
        if abs(d) <= tol and (best is None or abs(d) < abs(best[0])):
            best = (d, g)
    if best is None:
        return "в разметке нет"
    d, g = best
    return f"разметка {g['event']} {mmss(float(g['t']))} (Δ {d:+.0f} с) — {g.get('note', '')[:70]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--chat-env", default="TELEGRAM_CHAT_ID")
    ap.add_argument("--pause", default="3-5", help="пауза между сообщениями, с (диапазон)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    plan = yaml.safe_load(Path(a.plan).read_text(encoding="utf-8"))
    events, sessions, titles, gts = [], [], {}, {}
    for i, seg in enumerate(plan["segments"]):
        e, s = load_segment(seg, i)
        events += e
        sessions += s
        for z, title in seg["tables"].items():
            titles[f"s{i}:{z}"] = title
        if seg.get("gt"):
            gts[seg["name"]] = yaml.safe_load((ROOT / seg["gt"]).read_text(encoding="utf-8"))
    names = list(titles.values())
    assert len(names) == len(set(names)), f"номера столов в плане пересекаются: {names}"
    events.sort(key=lambda e: (e["ts"], e["id"]))

    def clock(t: float) -> str:
        t %= DAY_S
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}"

    feed = TelegramFeed(clock, titles=titles)
    actions = []
    for e in events:
        for act in feed.handle([e]):
            actions.append((act, e))
    end = max([e["ts"] for e in events] + [0.0])
    r = shift_report(sessions, events, clock, hour_of=lambda t: int(t % DAY_S // 3600), titles=titles, t_end=end)
    summary = shift_html(r, plan.get("site", "Зал · демо"), plan.get("day", ""))

    print(f"Сообщений: {sum(1 for x, _ in actions if x.op == 'send')} алертов, "
          f"{sum(1 for x, _ in actions if x.op == 'edit')} подтверждений правкой, + итог смены\n")
    for act, e in actions:
        src = e["src"]
        print(f"[{clock(act.ts)}] {act.op.upper():4} {titles[e['zone_id']]:7} ← {src['segment']} {src['zone']} "
              f"{e['type']} в {mmss(src['file_ts'])} файла")
        print("      " + act.html.replace("\n", "\n      "))
        gt = gts.get(src["segment"])
        typ = e["type"] if e["type"] != "alert" else {"no_waiter": "seated", "dirty_table": "left"}.get(
            e["payload"].get("kind"), "")
        ref_t = src["file_ts"] - float(e["payload"].get("waited_s") or e["payload"].get("dirty_s") or 0)
        print("      сверка: " + gt_near(gt, src["zone"], typ, ref_t if e["type"] == "alert" else src["file_ts"]))
    print("\nСессии в итоге (время сцены):")
    for s in sorted(sessions, key=lambda s: s["seated_ts"]):
        print(f"  {titles[s['zone_id']]:7} сели {clock(s['seated_ts'])}"
              + (" (до записи)" if s.get("preexisting") else "")
              + (f", подход через {s['response_time_s']:.0f} с" if s.get("response_time_s") else "")
              + (f", ушли {clock(s['left_ts'])}" if s.get("left_ts") else "")
              + (f", убрали через {s['cleanup_time_s']:.0f} с" if s.get("cleanup_time_s") else ""))
    print("\n" + summary)
    print("\nЦифры итога:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
    if not a.send:
        print("\n(не отправлено: добавьте --send)")
        return 0
    load_dotenv_if_any(ROOT)
    from qonaq.alerts.telegram import sender_from_env

    s = sender_from_env(a.chat_env)
    if s is None:
        print(f"нет TELEGRAM_BOT_TOKEN / {a.chat_env}")
        return 2
    lo, hi = (float(x) for x in a.pause.split("-"))
    mids: dict = {}
    for act, _ in actions:
        if act.op == "send":
            mids[act.key] = s.send_html(act.html, act.markup)
        elif act.key in mids:
            s.edit_html(mids[act.key], act.html, act.markup)
        print(f"  {act.op} {clock(act.ts)} — ок", flush=True)
        time.sleep(random.uniform(lo, hi))  # noqa: S311 — пауза для читаемости, не криптография
    s.send_html(summary)
    print("  итог смены — ок")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
