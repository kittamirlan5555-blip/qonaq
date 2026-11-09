"""Гость vs персонал: эвристики по поведению + форма + голосование по окну (SPEC §4, §9a п.3).

Не знает про видео: на вход — наблюдения людей (id трека, bbox, зона-стол, гистограмма торса),
на выход — роль трека с уверенностью. Роль не скачет от кадра к кадру (голосование + гистерезис).
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from qonaq.config import RolesConfig
from qonaq.roles.uniform import UniformProfile

GUEST, STAFF, UNKNOWN = "guest", "staff", "unknown"


@dataclass(frozen=True)
class PersonObs:
    track_id: int
    xyxy: tuple[float, float, float, float]
    table_zone: str | None = None
    torso_hist: np.ndarray | None = None


@dataclass(frozen=True)
class RoleDecision:
    role: str
    confidence: float
    standing: bool
    dwell_s: float
    reasons: tuple[str, ...] = ()
    stable_id: int = 0  # id, переживающий смену id трекером (наследуется от потерянного трека рядом)


@dataclass
class _TrackState:
    first_ts: float
    last_ts: float
    xyxy: tuple[float, float, float, float]
    zone: str | None = None
    zone_since: float = 0.0
    zone_last_seen: float = 0.0
    positions: deque = field(default_factory=deque)  # (ts, cx, cy)
    tables: dict[str, float] = field(default_factory=dict)  # зона -> когда последний раз был
    uniform: float | None = None  # сглаженная похожесть на форму
    votes: deque = field(default_factory=deque)  # (ts, role)
    role: str = UNKNOWN
    confidence: float = 0.0
    stable_id: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def diag(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return float(np.hypot(x2 - x1, y2 - y1))


def _noisy_or(scores: list[float]) -> float:
    p = 1.0
    for s in scores:
        p *= 1.0 - s
    return 1.0 - p


class RoleTracker:
    ZONE_EXIT_TOLERANCE_S = 2.0

    def __init__(self, cfg: RolesConfig | None = None, seat_min_s: float = 15.0,
                 uniform: UniformProfile | None = None):
        self.cfg = cfg or RolesConfig()
        self.seat_min_s = seat_min_s
        self.uniform = uniform
        self._tracks: dict[int, _TrackState] = {}

    # ------------------------------------------------------------------ state
    def _inherit(self, obs: PersonObs, ts: float, present: set[int]) -> _TrackState | None:
        """Новый id рядом с недавно потерянным треком наследует его историю (смена id у ByteTrack)."""
        x1, y1, x2, y2 = obs.xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best, best_d = None, float("inf")
        for tid, st in self._tracks.items():
            if tid in present or ts - st.last_ts > self.cfg.reid_gap_s or ts - st.last_ts <= 0:
                continue
            d = float(np.hypot(cx - st.center[0], cy - st.center[1]))
            if d < self.cfg.reid_dist * max(st.diag, 1.0) and d < best_d:
                best, best_d = tid, d
        if best is None:
            return None
        return self._tracks.pop(best)

    def update(self, ts: float, people: list[PersonObs]) -> dict[int, RoleDecision]:
        cfg = self.cfg
        present = {p.track_id for p in people}
        out: dict[int, RoleDecision] = {}
        for p in people:
            st = self._tracks.get(p.track_id)
            if st is None:
                st = self._inherit(p, ts, present) or _TrackState(first_ts=ts, last_ts=ts, xyxy=p.xyxy,
                                                                   stable_id=p.track_id)
                self._tracks[p.track_id] = st
            st.last_ts, st.xyxy = ts, p.xyxy
            # --- зона-стол с допуском на кратковременный выход
            if p.table_zone is not None:
                if p.table_zone != st.zone:
                    st.zone, st.zone_since = p.table_zone, ts
                st.zone_last_seen = ts
                st.tables[p.table_zone] = ts
            elif st.zone is not None and ts - st.zone_last_seen > self.ZONE_EXIT_TOLERANCE_S:
                st.zone = None
            # --- история позиций (центр bbox устойчивее нижней точки для сидящих)
            st.positions.append((ts, *st.center))
            horizon = max(self.seat_min_s, 1.0)
            while st.positions and st.positions[0][0] < ts - horizon:
                st.positions.popleft()
            # --- форма
            if self.uniform is not None and p.torso_hist is not None:
                u = self.uniform.score(p.torso_hist)
                st.uniform = u if st.uniform is None else 0.8 * st.uniform + 0.2 * u
            out[p.track_id] = self._decide(st, ts)
        # забываем давно пропавшие
        ttl = max(cfg.multi_table_window_s, 30.0)
        for tid in [t for t, st in self._tracks.items() if ts - st.last_ts > ttl]:
            del self._tracks[tid]
        return out

    # ------------------------------------------------------------------ decision
    def _instant(self, st: _TrackState, ts: float) -> tuple[str, float, tuple[str, ...], bool, float]:
        cfg = self.cfg
        x1, y1, x2, y2 = st.xyxy
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        standing = h / w >= cfg.standing_aspect
        dwell = ts - st.zone_since if st.zone is not None else 0.0
        guest_scores, staff_scores, reasons = [], [], []
        # 1) долго в зоне стола и почти не двигается -> гость
        if st.zone is not None and dwell >= self.seat_min_s and len(st.positions) >= 3:
            pts = np.array([(x, y) for _, x, y in st.positions], dtype=np.float32)
            med = np.median(pts, axis=0)
            spread = float(np.percentile(np.hypot(pts[:, 0] - med[0], pts[:, 1] - med[1]), 90))
            if spread < cfg.move_threshold * h:
                guest_scores.append(0.7 if standing else 0.8)
                reasons.append("dwell")
        # 2) побывал у >= N разных столов за окно -> персонал (усиливающий признак)
        recent = [z for z, t in st.tables.items() if ts - t <= cfg.multi_table_window_s]
        if len(recent) >= cfg.multi_table_count:
            staff_scores.append(0.6)
            reasons.append("multi_table")
        # 3) форма
        if self.uniform is not None and st.uniform is not None:
            thr = self.uniform.threshold
            if st.uniform >= thr:
                staff_scores.append(min(0.95, 0.6 + 0.35 * (st.uniform - thr) / max(1e-6, 1 - thr)))
                reasons.append("uniform")
        g, s = _noisy_or(guest_scores), _noisy_or(staff_scores)
        if s >= 0.5 and s >= g:
            return STAFF, s, tuple(reasons), standing, dwell
        if g >= 0.5:
            return GUEST, g, tuple(reasons), standing, dwell
        return UNKNOWN, 1.0 - max(g, s), tuple(reasons), standing, dwell

    def _decide(self, st: _TrackState, ts: float) -> RoleDecision:
        role, conf, reasons, standing, dwell = self._instant(st, ts)
        st.votes.append((ts, role, conf))
        while st.votes and st.votes[0][0] < ts - self.cfg.vote_window_s:
            st.votes.popleft()
        counts = Counter(v[1] for v in st.votes)
        top, n = counts.most_common(1)[0]
        frac = n / len(st.votes)
        if top != st.role:
            # Гистерезис: в «известную» роль — при 60% голосов; обратно в unknown — только при 90%.
            need = 0.9 if top == UNKNOWN else 0.6
            if frac >= need:
                st.role = top
        confs = [v[2] for v in st.votes if v[1] == st.role]
        st.confidence = float(np.mean(confs)) if confs else 0.3
        return RoleDecision(st.role, round(st.confidence, 2), standing, round(dwell, 1), reasons, st.stable_id)

    def tables_visited(self, track_id: int, ts: float) -> int:
        st = self._tracks.get(track_id)
        if st is None:
            return 0
        return sum(1 for t in st.tables.values() if ts - t <= self.cfg.multi_table_window_s)
