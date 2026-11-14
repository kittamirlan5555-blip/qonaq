"""B2: профиль billiard — чистая логика на синтетических таймлайнах (без кадров)."""
from __future__ import annotations

from qonaq.config import BilliardThresholds, SiteConfig
from qonaq.logic.base import FeltSignal, ZoneObservation
from qonaq.logic.billiard import BilliardEngine, BilliardTableMachine, BState
from qonaq.zones.models import Zone, ZoneSet

FPS = 2.0


class TL:
    """Таймлайн одного стола: по умолчанию лампа горит, движения нет."""

    def __init__(self, duration_s: float, lamp: bool = True):
        n = int(duration_s * FPS)
        self.ts = [i / FPS for i in range(n)]
        self.sig = [dict(motion=False, lamp_on=lamp, covered=False) for _ in range(n)]

    def set(self, a: float, b: float, **kw):
        for i, t in enumerate(self.ts):
            if a <= t < b:
                self.sig[i].update(kw)
        return self

    def shot(self, t: float, dur: float = 4.0, intrusion_before: float = 2.0):
        """Удар: игрок наклонился (крупный объект через край сукна), потом шары катятся dur секунд."""
        self.set(t - intrusion_before, t + 0.5, intrusion=True)
        return self.set(t, t + dur, motion=True)

    def game(self, start: float, end: float, every: float = 40.0, dur: float = 4.0):
        t = start
        while t < end:
            self.shot(t, dur)
            t += every
        return self

    def drop(self, a: float, b: float):
        """Обрыв камеры: наблюдений нет."""
        keep = [(t, s) for t, s in zip(self.ts, self.sig, strict=True) if not (a <= t < b)]
        self.ts, self.sig = [k[0] for k in keep], [k[1] for k in keep]
        return self

    def run(self, m: BilliardTableMachine, finish: bool = True):
        out = []
        for t, s in zip(self.ts, self.sig, strict=True):
            out += m.update(ZoneObservation(ts=t, zone_id=m.zone_id, felt=FeltSignal(**s)))
        if finish:
            out += m.finish(self.ts[-1])
        return out


def games(events):
    return [e.payload for e in events if e.type == "game_ended"]


def th(**kw):
    return BilliardThresholds(**kw)


def test_normal_game_start_end_retroactive():
    tl = TL(3600).game(300, 2400, every=45)
    evs = tl.run(BilliardTableMachine("B1", th()))
    g = games(evs)
    assert len(g) == 1
    assert g[0]["start"] == 300.0  # задним числом — с первого удара
    last_shot = max(t for t in range(300, 2400, 45))
    assert abs(g[0]["end"] - (last_shot + 4.0)) <= 1.0  # конец — по последнему эпизоду, не по P
    assert g[0]["reason"] == "pause"
    started = [e for e in evs if e.type == "game_started"]
    assert started[0].ts == 300.0 and started[0].payload["detected_ts"] < 300 + 3 * 45 + 10


def test_neighbour_walking_past_right_table_is_not_a_game():
    """Сосед играет на левом столе и проходит у правого: у правого только крупные объекты через край."""
    left = TL(1800).game(100, 1500)
    right = TL(1800)
    for t in range(120, 1500, 60):  # проходит/стоит у правого стола, наклоняется за мелом
        right.set(t, t + 6, intrusion=True)
    evs_l = left.run(BilliardTableMachine("B1"))
    evs_r = right.run(BilliardTableMachine("B2"))
    assert len(games(evs_l)) == 1 and games(evs_r) == []


def test_single_shot_between_moves_is_not_a_game():
    """Игрок соседнего стола «между делом» толкнул шары на втором столе — один-два эпизода за 5 минут."""
    tl = TL(1800).shot(200).shot(900).shot(1500)  # три удара, но не в одном окне T=300 с
    assert games(tl.run(BilliardTableMachine("B2"))) == []


def test_brushing_felt_and_flicker_blips_are_not_episodes():
    """Протирка сукна: рука со щёткой (крупное через край) + одиночные кадры «движения»; мерцание ламп."""
    tl = TL(1200).set(100, 400, intrusion=True)
    for t in range(100, 400, 7):
        tl.set(t, t + 0.5, motion=True)  # одиночные кадры — не эпизоды (episode_min_frames=2)
    for t in range(500, 900, 5):
        tl.set(t, t + 3, motion=True, global_change=True)  # свет мигает — кадры нейтральны
    assert games(tl.run(BilliardTableMachine("B1"))) == []


def test_covering_table_ends_session_and_blocks_new_one():
    tl = TL(2400).game(100, 1000).set(1100, 2400, covered=True)
    tl.set(1200, 1210, motion=True)  # кто-то поправляет покрывало
    g = games(tl.run(BilliardTableMachine("B1")))
    assert len(g) == 1 and g[0]["reason"] == "covered"


def test_long_pause_splits_short_pause_does_not():
    m = BilliardTableMachine("B1", th(idle_s=240, pause_max_s=600))
    tl = TL(5400).game(100, 1000).game(1400, 2000).game(2900, 3600)  # паузы ~6 мин и ~15 мин
    evs = tl.run(m)
    g = games(evs)
    assert len(g) == 2
    assert g[0]["start"] == 100.0 and g[0]["pauses"] and 300 <= g[0]["pauses"][0] <= 420
    assert g[1]["start"] == 2900.0
    assert any(e.type == "state_change" and e.payload["to"] == "PAUSED" for e in evs)


def test_light_off_ends_session_at_last_shot():
    tl = TL(2400).game(100, 1200).set(1300, 2400, lamp_on=False)
    tl.set(1295, 1305, global_change=True)
    evs = tl.run(BilliardTableMachine("B1"))
    g = games(evs)
    assert len(g) == 1 and g[0]["reason"] == "lamp_off"
    assert g[0]["end"] < 1200 + 5  # конец — последний удар, а не момент выключения
    kinds = [e.type for e in evs]
    assert "lamp_off" in kinds and kinds.index("lamp_off") < kinds.index("game_ended")


def test_lamp_on_without_play_is_reported_but_not_a_game():
    tl = TL(1200, lamp=False).set(300, 1200, lamp_on=True)
    evs = tl.run(BilliardTableMachine("B1"))
    assert games(evs) == [] and [e.type for e in evs if e.type.startswith("lamp")] == ["lamp_on"]


def test_camera_outage_freezes_pause_timer():
    """Обрыв 12 минут посреди игры (> P = 10 мин) не закрывает сессию задним числом."""
    tl = TL(4000).game(100, 1500).drop(1520, 2240).game(2260, 3000)
    g = games(tl.run(BilliardTableMachine("B1")))
    assert len(g) == 1 and g[0]["blind_s"] >= 700


def test_flicker_in_detection_still_forms_episodes():
    """Детекция мигает: внутри удара пропадает каждый третий кадр — эпизод не рвётся (gap ≤ 3 с)."""
    tl = TL(1800).game(100, 1200)
    for i, _t in enumerate(tl.ts):
        if i % 3 == 0:
            tl.sig[i]["motion"] = False
    assert len(games(tl.run(BilliardTableMachine("B1")))) == 1


def test_engine_mixes_billiard_tables_and_bar_seats():
    zs = ZoneSet(camera_id="c", zones=[
        Zone(id="B1", type="billiard_table", polygon=[(0, 0), (100, 0), (100, 50), (0, 50)]),
        Zone(id="T1", type="table", polygon=[(200, 0), (240, 0), (240, 40), (200, 40)])])
    eng = BilliardEngine(zs, SiteConfig(profile="billiard"))
    tl = TL(900).game(60, 700, every=30)
    evs = []
    for t, s in zip(tl.ts, tl.sig, strict=True):
        evs += eng.update([ZoneObservation(ts=t, zone_id="B1", felt=FeltSignal(**s)),
                           ZoneObservation(ts=t, zone_id="T1", guest_tracks=(1,) if t > 100 else ())])
    assert any(e.type == "game_started" for e in evs) and any(e.type == "seated" for e in evs)
    snap = eng.snapshot(tl.ts[-1])
    assert snap["B1"]["state"] in (BState.IN_PLAY.value, BState.PAUSED.value) and snap["B1"]["label"].startswith("игра")
    assert eng.summary(tl.ts[-1])["in_play"] == 1


# ---------------------------------------------------------------- видео-слой сукна на процедурной синтетике
def _felt_run(timeline, duration=40, flicker=0.03, view="angled", tables=None):
    from qonaq.billiard.felt import FeltAnalyzer
    from qonaq.billiard.synth import Scene

    spec = {"name": "t", "fps": 2, "duration_s": duration, "view": view, "flicker": flicker, "seed": 1,
            "tables": tables or [{"id": "B1", "origin": [0.9, 0.9]}, {"id": "B2", "origin": [4.4, 0.9], "felt": "blue"}],
            "timeline": timeline}
    sc = Scene(spec)
    an = {z["id"]: FeltAnalyzer(z["polygon"], sc.size) for z in sc.zones()["zones"]}
    sig = {k: [] for k in an}
    for t, f in sc.frames():
        for k, a in an.items():
            sig[k].append((t, a.update(f)))
    return sc, sig


def _frames(sig, a, b, pred):
    return sum(1 for t, s in sig if a <= t < b and pred(s))


def test_felt_shots_give_motion_and_neighbour_lean_gives_only_intrusion():
    sc, sig = _felt_run([{"table": "B1", "game": [5, 40], "every": [10, 12], "seed": 2},
                         {"table": "B2", "lean": [5, 40]}, {"walk": [[0, 0.2, 2.55], [40, 7.8, 2.55]]}])
    moving = [(a, b) for a, b in sc.ball_moving["B1"] if b - a > 2]
    assert moving
    for a, b in moving:  # шары катятся — движение видно хотя бы на половине кадров
        n = int((b - a) * 2)
        assert _frames(sig["B1"], a + 0.5, b, lambda s: s.motion) >= n // 2
    assert _frames(sig["B2"], 0, 40, lambda s: s.motion) <= 1
    assert _frames(sig["B2"], 5, 40, lambda s: s.intrusion) >= 30


def test_felt_brushing_is_intrusion_not_balls():
    _, sig = _felt_run([{"table": "B2", "brush": [5, 40]}])
    runs, cur = [], 0
    for _, s in sig["B2"]:
        cur = cur + 1 if s.motion else 0
        runs.append(cur)
    assert max(runs) <= 1  # нет двух кадров «шаров» подряд — эпизода нет


def test_felt_lamp_off_global_change_and_cover():
    _, sig = _felt_run([{"table": "B1", "lamp": [[0, 20]]}, {"table": "B2", "cover": [20, 40]}], flicker=0.0)
    b1 = sig["B1"]
    assert all(s.lamp_on for t, s in b1 if 5 <= t < 19.5) and all(s.lamp_on is False for t, s in b1 if t >= 21)
    assert any(s.global_change for t, s in b1 if 19.5 <= t <= 21)
    assert _frames(b1, 0, 40, lambda s: s.motion) == 0
    b2 = sig["B2"]
    assert all(s.covered is False for t, s in b2 if 5 <= t < 19.5) and all(s.covered for t, s in b2 if t >= 21)


def test_felt_flicker_and_room_lights_do_not_move_balls():
    _, sig = _felt_run([{"lights_off": [10, 25]}], flicker=0.06, view="top")
    for k in ("B1", "B2"):
        assert _frames(sig[k], 0, 40, lambda s: s.motion) == 0
