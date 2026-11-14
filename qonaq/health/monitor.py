"""Проверки здоровья (W7). Не знает про видео-источник: на вход — кадры/результаты и события, на выход — проблемы.

Проверки:
- camera_down — кадров нет дольше `stale_s` (только для потоков, по wall clock);
- dark_frame — средняя яркость ниже `dark_luma` дольше `dark_s`;
- frozen_frame — поток «завис» дольше `frozen_s` (B1): нет ни движения, ни шума матрицы/сжатия по всему кадру.
  Неподвижная живая сцена (пустой зал ночью) — НЕ проблема: у живой камеры между кадрами меняются отдельные
  пиксели по всему кадру (шум сенсора, I-кадры кодека), это видно на полном разрешении, а не на уменьшенном кадре.
  Если меняется только маленький участок (идут часы OSD регистратора), а картинка стоит — тоже frozen_frame,
  с пояснением «время на картинке идёт» (камера зависла, регистратор жив). Неподвижность — поле `still_s` в status;
- flapping — стол переключал состояние чаще `flap_per_hour` раз за последний час;
- long_wait — ожидание официанта > `max_wait_s` (40 мин) — скорее ошибка, чем реальность;
- short_sessions — `short_session_run` сессий подряд короче `short_session_s` на одном столе;
- no_staff_visits — за смену (≥ `shift_min_s` работы и ≥ 3 посадок) ни одного визита персонала;
- unknown_share — доля наблюдений людей с ролью unknown за окно > `unknown_max`.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from qonaq.logic.base import Event


@dataclass
class HealthConfig:
    stale_s: float = 30.0
    dark_luma: float = 18.0
    dark_s: float = 60.0
    # «в кадре что-то движется»: на уменьшенном 64×36 кадре ≥ motion_frac пикселей поменялись ≥ motion_px уровней.
    # (Раньше — средняя |разница| < 0,25; на целых 0..255 шум σ=1 уже даёт ~0,4, а тусклый пустой зал — меньше 0,25,
    # отсюда ложное «завис» в пустом зале и на синтетике.)
    motion_px: float = 6.0
    motion_frac: float = 0.01  # ≥ 23 из 2304; смена минут в часах OSD — 3–6 пикселей, человек — десятки
    frozen_s: float = 120.0
    # B1: «новый кадр» (живая камера в неподвижной сцене): кадр, поменявшийся относительно предыдущего
    # (> refresh_cells клеток), не похож ни на один кадр за последние novel_history_s (≥ novel_cells клеток
    # отличаются от каждого). Живая камера на каждом I-кадре кодека приносит новую реализацию шума матрицы;
    # зависшая камера, которую кодек пережимает, повторяет одни и те же кадры с периодом GOP (EVAL §18.1).
    refresh_cells: float = 0.05
    novel_cells: float = 0.5
    novel_history_s: float = 60.0
    novel_min_history_s: float = 20.0  # раньше не решаем (нечего сравнивать)
    osd_cells: float = 0.05  # меняется ≤ 5% клеток (часы OSD) — пояснение «время на картинке идёт»
    flap_per_hour: int = 12
    max_wait_s: float = 2400.0
    short_session_s: float = 60.0
    short_session_run: int = 3
    shift_min_s: float = 4 * 3600.0
    unknown_max: float = 0.5
    unknown_window_s: float = 600.0
    unknown_min_obs: int = 200
    # S2: компьютер не успевает — обработанных кадров/с за окно меньше доли от process_fps (только live)
    target_fps: float | None = None
    slow_ratio: float = 0.75
    slow_window_s: float = 300.0


@dataclass
class Issue:
    kind: str
    camera_id: str
    zone_id: str = ""
    severity: str = "warn"  # warn | error
    message: str = ""
    since: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MESSAGES = {
    "camera_down": "камера не отвечает {s:.0f} с",
    "camera_recovered": "камера снова работает (не было {s:.0f} с)",
    "dark_frame": "кадр тёмный {s:.0f} с (свет выключен или камера закрыта)",
    "frozen_frame": "кадр не меняется {s:.0f} с: кадры одинаковые, нет даже шума матрицы — поток завис "
                    "(перезапустите камеру/канал на регистраторе)",
    "frozen_frame_osd": "изображение стоит {s:.0f} с, хотя время на картинке идёт — камера зависла, регистратор "
                        "отвечает (перезапустите камеру по питанию/PoE)",
    "flapping": "стол {zone} переключался {n} раз за час — проверьте зону",
    "long_wait": "стол {zone}: ожидание {m:.0f} мин — неправдоподобно, проверьте",
    "short_sessions": "стол {zone}: {n} сессий подряд короче {s:.0f} с — детекция мигает?",
    "no_staff_visits": "за смену ни одного визита персонала при {n} посадках — роли не работают?",
    "unknown_share": "роль не определена у {p:.0f}% наблюдений людей",
    "slow_processing": "обрабатывается {f:.1f} кадр/с вместо {t:g} — компьютер не успевает (снизьте process_fps, "
                       "detect_every 2 или профиль детектора полегче; EVAL §15)",
}


def _noise_view(gray: np.ndarray, max_w: int = 160) -> np.ndarray:
    """Прореживание через шаг без усреднения (усреднение INTER_AREA съедает шум матрицы и артефакты кодека)."""
    step = max(1, int(np.ceil(gray.shape[1] / max_w)))
    return np.ascontiguousarray(gray[::step, ::step])


def live_cells_fraction(cur: np.ndarray, prev: np.ndarray, noise_px: int = 2, cell_px: float = 0.01,
                        grid: tuple[int, int] = (16, 9)) -> float:
    """Доля клеток сетки, где поменялось ≥ cell_px пикселей хотя бы на noise_px уровней яркости.

    Новая реализация шума матрицы (I-кадр живой камеры) — почти все клетки. Часы OSD — 1–3 клетки из 144.
    """
    d = np.abs(cur.astype(np.int16) - prev.astype(np.int16)) >= noise_px
    gx, gy = grid
    h, w = d.shape
    hh, ww = h // gy * gy, w // gx * gx
    if hh == 0 or ww == 0:
        return 1.0 if d.any() else 0.0
    cells = d[:hh, :ww].reshape(gy, hh // gy, gx, ww // gx).mean(axis=(1, 3))
    return float((cells >= cell_px).mean())


class _Cam:
    def __init__(self) -> None:
        self.last_wall: float | None = None
        self.prev_small: np.ndarray | None = None
        self.dark_since: float | None = None
        self.frozen_since: float | None = None  # B1: с какого момента нет ни движения, ни новых кадров
        self.still_since: float | None = None  # нет движения (сцена неподвижна)
        self.osd_n = 0  # кадров с маленьким изменением (часы OSD) с frozen_since
        self.frozen_n = 0
        self.last_novel: float | None = None
        self.hist: deque = deque()  # (ts, прореженный кадр) за novel_history_s
        self.state_changes: dict[str, deque] = defaultdict(deque)
        self.short_run: dict[str, int] = defaultdict(int)
        self.waiting_since: dict[str, float] = {}
        self.roles: deque = deque()  # (ts, n_unknown, n_total)
        self.first_ts: float | None = None
        self.last_ts: float = 0.0
        self.seated = 0
        self.visits = 0
        self.seated_ts: dict[str, float] = {}
        self.outages: list[tuple[float, float]] = []  # (начало, конец) по wall clock — уже закончившиеся обрывы
        self.frames = 0
        self.walls: deque = deque()  # wall clock обработанных кадров за slow_window_s


class HealthMonitor:
    def __init__(self, cfg: HealthConfig | None = None, clock=time.time):
        self.cfg = cfg or HealthConfig()
        self.clock = clock
        self._cams: dict[str, _Cam] = defaultdict(_Cam)
        self.flip_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ входы
    def register(self, camera_id: str) -> None:
        """Камера ожидается: если от неё не придёт ни одного кадра за stale_s — это тоже camera_down."""
        c = self._cams[camera_id]
        if c.last_wall is None:
            c.last_wall = self.clock()

    def observe_frame(self, camera_id: str, ts: float, frame: np.ndarray | None) -> None:
        c = self._cams[camera_id]
        now = self.clock()
        if c.last_wall is not None and now - c.last_wall > self.cfg.stale_s:
            c.outages.append((c.last_wall, now))  # камера вернулась после обрыва — запоминаем эпизод
        c.last_wall = now
        c.frames += 1
        c.walls.append(now)
        while c.walls and c.walls[0] < now - self.cfg.slow_window_s:
            c.walls.popleft()
        c.first_ts = ts if c.first_ts is None else c.first_ts
        c.last_ts = ts
        if frame is None:
            return
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        luma = float(small.mean())
        if luma < self.cfg.dark_luma:
            c.dark_since = ts if c.dark_since is None else c.dark_since
        else:
            c.dark_since = None
        view = _noise_view(g)
        if c.prev_small is not None:
            moving = float((np.abs(small - c.prev_small) >= self.cfg.motion_px).mean()) >= self.cfg.motion_frac
            c.still_since = None if moving else (ts if c.still_since is None else c.still_since)
            live = moving or self._novel(c, ts, view)
            if live:
                c.frozen_since, c.osd_n, c.frozen_n = None, 0, 0
            else:
                c.frozen_since = ts if c.frozen_since is None else c.frozen_since
                c.frozen_n += 1
                if c.hist and c.hist[-1][1].shape == view.shape:
                    f = live_cells_fraction(view, c.hist[-1][1])
                    c.osd_n += int(0.0 < f <= self.cfg.osd_cells)
        c.hist.append((ts, view))
        while c.hist and c.hist[0][0] < ts - self.cfg.novel_history_s:
            c.hist.popleft()
        c.prev_small = small

    def _novel(self, c: _Cam, ts: float, view: np.ndarray) -> bool:
        """Камера жива в неподвижной сцене? Да, если недавно был кадр, не похожий ни на один за минуту."""
        cfg = self.cfg
        if c.last_novel is not None and ts - c.last_novel < 10.0:
            return True  # недавно подтвердили — не тратим CPU
        if not c.hist or c.hist[-1][1].shape != view.shape or ts - c.hist[0][0] < cfg.novel_min_history_s:
            return False
        if live_cells_fraction(view, c.hist[-1][1]) <= cfg.refresh_cells:
            return False
        past = list(c.hist)[:-2]
        if not past or any(live_cells_fraction(view, h) < cfg.novel_cells for _, h in reversed(past)):
            return False
        c.last_novel = ts
        return True

    def observe_roles(self, camera_id: str, ts: float, roles: list[str]) -> None:
        c = self._cams[camera_id]
        if roles:
            c.roles.append((ts, sum(1 for r in roles if r not in ("guest", "staff")), len(roles)))
        while c.roles and c.roles[0][0] < ts - self.cfg.unknown_window_s:
            c.roles.popleft()

    def observe_events(self, camera_id: str, events: list[Event]) -> None:
        c = self._cams[camera_id]
        for e in events:
            z = e.zone_id
            if e.type == "state_change":
                dq = c.state_changes[z]
                dq.append(e.ts)
                while dq and dq[0] < e.ts - 3600:
                    dq.popleft()
                to = e.payload.get("to")
                if to == "SEATED_WAITING":
                    c.waiting_since[z] = e.ts
                elif z in c.waiting_since:
                    c.waiting_since.pop(z, None)
            elif e.type in ("seated", "seated_at_dirty_table"):
                if e.type == "seated":
                    c.seated += 1
                c.seated_ts[z] = e.ts
            elif e.type in ("served", "staff_visit"):
                c.visits += 1
            elif e.type in ("left", "walkout_before_service"):
                dur = e.payload.get("visit_duration_s")
                if dur is None and z in c.seated_ts:
                    dur = e.ts - c.seated_ts[z]
                if dur is not None and dur < self.cfg.short_session_s:
                    c.short_run[z] += 1
                else:
                    c.short_run[z] = 0

    def observe_result(self, r) -> None:
        """FrameResult пайплайна целиком."""
        self.observe_frame(r.camera_id, r.ts, r.frame)
        self.observe_roles(r.camera_id, r.ts, [p.role for p in r.people])
        self.observe_events(r.camera_id, r.events)

    # ------------------------------------------------------------------ проверки
    def check(self, live: bool = True) -> list[Issue]:
        cfg = self.cfg
        out: list[Issue] = []
        now = self.clock()
        for cid, c in self._cams.items():
            if live and c.last_wall is not None and now - c.last_wall > cfg.stale_s:
                out.append(self._issue("camera_down", cid, "error", s=now - c.last_wall))
            if live and cfg.target_fps and c.walls:
                span = now - c.walls[0]
                recent_outage = any(b > now - cfg.slow_window_s for _, b in c.outages)
                if span >= 0.9 * cfg.slow_window_s and not recent_outage and now - c.last_wall <= cfg.stale_s:
                    fps = len(c.walls) / span
                    if fps < cfg.slow_ratio * cfg.target_fps:
                        out.append(self._issue("slow_processing", cid, f=fps, t=cfg.target_fps))
            if c.dark_since is not None and c.last_ts - c.dark_since >= cfg.dark_s:
                out.append(self._issue("dark_frame", cid, s=c.last_ts - c.dark_since))
            if c.frozen_since is not None and c.last_ts - c.frozen_since >= cfg.frozen_s:
                osd = c.frozen_n > 0 and c.osd_n >= 0.5 * c.frozen_n
                iss = self._issue("frozen_frame_osd" if osd else "frozen_frame", cid, "error",
                                  s=c.last_ts - c.frozen_since)
                iss.kind = "frozen_frame"  # для подписчиков вид один, различается текстом и data.osd
                iss.data["osd"] = osd
                out.append(iss)
            for z, dq in c.state_changes.items():
                n = sum(1 for t in dq if t >= c.last_ts - 3600)
                if n > cfg.flap_per_hour:
                    out.append(self._issue("flapping", cid, zone=z, n=n))
            for z, since in c.waiting_since.items():
                if c.last_ts - since > cfg.max_wait_s:
                    out.append(self._issue("long_wait", cid, zone=z, m=(c.last_ts - since) / 60))
            for z, n in c.short_run.items():
                if n >= cfg.short_session_run:
                    out.append(self._issue("short_sessions", cid, zone=z, n=n, s=cfg.short_session_s))
            worked = c.last_ts - (c.first_ts if c.first_ts is not None else c.last_ts)
            if worked >= cfg.shift_min_s and c.seated >= 3 and c.visits == 0:
                out.append(self._issue("no_staff_visits", cid, n=c.seated))
            n_obs = sum(x[2] for x in c.roles)
            if n_obs >= cfg.unknown_min_obs:
                share = sum(x[1] for x in c.roles) / n_obs
                if share > cfg.unknown_max:
                    out.append(self._issue("unknown_share", cid, p=100 * share))
        return out

    def _issue(self, kind: str, cid: str, severity: str = "warn", zone: str = "", **kw) -> Issue:
        return Issue(kind, cid, zone, severity, MESSAGES[kind].format(zone=zone, **{"s": 0, "n": 0, "m": 0, "p": 0,
                                                                                    "f": 0, "t": 0, **kw}), data=kw)

    def status(self, live: bool = True) -> dict[str, Any]:
        issues = self.check(live)
        sev = "error" if any(i.severity == "error" for i in issues) else ("warn" if issues else "ok")
        cams = {cid: {"last_ts": round(c.last_ts, 1), "seated": c.seated, "visits": c.visits, "frames": c.frames,
                      "unknown_share": round(sum(x[1] for x in c.roles) / max(1, sum(x[2] for x in c.roles)), 3),
                      "still_s": round(c.last_ts - c.still_since, 1) if c.still_since is not None else 0.0,
                      "outages": len(c.outages), "down_s_total": round(sum(b - a for a, b in c.outages), 1),
                      "last_outage": ({"start": round(c.outages[-1][0], 1), "end": round(c.outages[-1][1], 1),
                                       "s": round(c.outages[-1][1] - c.outages[-1][0], 1)} if c.outages else None)}
                for cid, c in self._cams.items()}
        return {"status": sev, "issues": [i.to_dict() for i in issues], "cameras": cams}


def check_events(events: list[dict], cfg: HealthConfig | None = None) -> list[Issue]:
    """Офлайн-проверка по журналу events.sqlite (без кадров): flapping, long_wait, short_sessions, no_staff_visits."""
    mon = HealthMonitor(cfg, clock=lambda: 0.0)
    by_cam: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_cam[e.get("camera_id") or "cam"].append(Event(e["ts"], e["zone_id"], e["type"], e.get("payload") or {},
                                                         e.get("camera_id") or "cam"))
    for cid, evs in by_cam.items():
        evs.sort(key=lambda e: e.ts)
        if evs:
            mon.observe_frame(cid, evs[0].ts, None)
        for e in evs:
            mon.observe_events(cid, [e])
            mon._cams[cid].last_ts = max(mon._cams[cid].last_ts, e.ts)
    return mon.check(live=False)
