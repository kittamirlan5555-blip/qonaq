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
    appearance: float | None = None  # p(staff) от RoleClassifier на этом кадре (None — не считали)


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
    app: deque = field(default_factory=deque)  # (ts, logit p(staff)) — внешний вид по треку
    hist: np.ndarray | None = None  # сглаженная гистограмма торса (для проверки подмены id)
    votes: deque = field(default_factory=deque)  # (ts, role)
    role: str = UNKNOWN
    confidence: float = 0.0
    stable_id: int = 0
    # Подмена id трекером (постановка 15.11): с какого момента одежда на треке устойчиво не та; последняя рамка и
    # время, когда на треке был «свой» человек, — по ним историю подхватит новый id на старом месте.
    mismatch_since: float | None = None
    sit_since: float | None = None  # с какого момента сидит без перерыва (None — стоит)
    stand_since: float | None = None
    good_xyxy: tuple[float, float, float, float] | None = None
    good_ts: float | None = None

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
        self._next_sid = 10_000_000  # стабильные id для «новых людей на старом id трекера»
        self._next_orphan = -1  # ключи «осиротевших» историй (id трекера ушёл к другому человеку)
        # Высота сидящих гостей по зонам (постановка 15.11): порог «стоит» подстраивается под камеру.
        self._seated_h: dict[str, deque] = {}

    # ------------------------------------------------------------------ state
    def _inherit(self, obs: PersonObs, ts: float, present: set[int]) -> _TrackState | None:
        """Новый id на месте недавно пропавшего человека наследует его историю (смена id у ByteTrack).

        Одежда (гистограмма торса) решает, кому: явно другая одежда — не наследует никогда (официант, вставший на
        место гостя, не получает «сидел 3 минуты»); та же одежда — наследует и после долгого перекрытия
        (`reid_clothes_gap_s`: официант на 5–10 с закрыл гостя собой). Без гистограммы — как раньше: только
        короткий разрыв `reid_gap_s`.
        """
        from qonaq.roles.uniform import similarity

        cfg = self.cfg
        x1, y1, x2, y2 = obs.xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best, best_key = None, None
        for tid, st in self._tracks.items():
            gap = ts - st.last_ts
            if tid in present or gap <= 0 or gap > max(cfg.reid_gap_s, cfg.reid_clothes_gap_s):
                continue
            d = float(np.hypot(cx - st.center[0], cy - st.center[1]))
            if d >= cfg.reid_dist * max(st.diag, 1.0):
                continue
            sim = similarity(st.hist, obs.torso_hist) if st.hist is not None and obs.torso_hist is not None else None
            if sim is not None and sim < cfg.identity_jump_similarity:
                continue
            if gap > cfg.reid_gap_s and (sim is None or sim < cfg.reid_clothes_similarity):
                continue
            key = (-(sim if sim is not None else 0.0), d)
            if best_key is None or key < best_key:
                best, best_key = tid, key
        if best is None:
            return None
        return self._tracks.pop(best)

    def _identity_switched(self, st: _TrackState, p: PersonObs, ts: float) -> bool:
        """Трекер перекинул id на другого человека? Резкая смена формы рамки и одежды — сразу (`_identity_jump`);
        одежда устойчиво другая `identity_switch_s` — тоже (прохожий заслонил сидящего, форма рамки почти та же)."""
        if self._identity_jump(st, p):
            return True
        if p.torso_hist is None or st.hist is None or not self.cfg.identity_switch_s:
            st.mismatch_since = None
            return False
        from qonaq.roles.uniform import similarity

        if similarity(st.hist, p.torso_hist) < self.cfg.identity_switch_similarity:
            if st.mismatch_since is None:
                st.mismatch_since = ts
            return ts - st.mismatch_since >= self.cfg.identity_switch_s
        st.mismatch_since = None
        return False

    def _detach(self, track_id: int) -> None:
        """История человека отцепляется от id трекера (id ушёл к другому) и ждёт, пока её подхватит новый id
        на старом месте (`_inherit`). Место и время — последние, когда на треке был он сам."""
        st = self._tracks.pop(track_id)
        if st.good_xyxy is not None:
            st.xyxy = st.good_xyxy
            st.last_ts = st.good_ts if st.good_ts is not None else st.last_ts
            while st.positions and st.positions[-1][0] > st.last_ts:
                st.positions.pop()
        st.mismatch_since = None
        self._tracks[self._next_orphan] = st
        self._next_orphan -= 1

    def _free_sid(self, track_id: int) -> int:
        """Стабильный id нового трека: его id трекера, если этот номер уже не носит другой (унаследованный) трек.
        Иначе два человека получили бы один стабильный id и логика стола считала бы их одним."""
        if any(st.stable_id == track_id for st in self._tracks.values()):
            self._next_sid += 1
            return self._next_sid
        return track_id

    def _identity_jump(self, st: _TrackState, p: PersonObs) -> bool:
        """Резкая смена пропорций рамки И другая одежда (гистограмма торса) → другой человек.
        Гость, вставший из-за стола, меняет пропорции, но не одежду — его история сохраняется."""
        cfg = self.cfg
        if not cfg.identity_jump_aspect or p.torso_hist is None or st.hist is None:
            return False
        x1, y1, x2, y2 = st.xyxy
        a1 = max(y2 - y1, 1.0) / max(x2 - x1, 1.0)
        x1, y1, x2, y2 = p.xyxy
        a2 = max(y2 - y1, 1.0) / max(x2 - x1, 1.0)
        if max(a1, a2) / min(a1, a2) < cfg.identity_jump_aspect:
            return False
        from qonaq.roles.uniform import similarity

        return similarity(st.hist, p.torso_hist) < cfg.identity_jump_similarity

    def update(self, ts: float, people: list[PersonObs]) -> dict[int, RoleDecision]:
        cfg = self.cfg
        present = {p.track_id for p in people}
        out: dict[int, RoleDecision] = {}
        # Сначала — подмены id: трекер отдал id другому человеку (официант/прохожий перекрыл гостя). Историю
        # (сидел N минут) новому человеку не отдаём; она ждёт своего хозяина — его новый id на старом месте.
        for p in people:
            st = self._tracks.get(p.track_id)
            if st is not None and self._identity_switched(st, p, ts):
                self._detach(p.track_id)
        for p in people:
            st = self._tracks.get(p.track_id)
            if st is None:
                st = self._inherit(p, ts, present) or _TrackState(first_ts=ts, last_ts=ts, xyxy=p.xyxy,
                                                                   stable_id=self._free_sid(p.track_id))
                self._tracks[p.track_id] = st
            st.last_ts, st.xyxy = ts, p.xyxy
            if st.mismatch_since is None:  # на треке «свой» человек: запоминаем место и одежду
                st.good_xyxy, st.good_ts = p.xyxy, ts
                if p.torso_hist is not None:
                    st.hist = p.torso_hist if st.hist is None else 0.8 * st.hist + 0.2 * p.torso_hist
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
            # --- сколько сидит без перерыва
            bw, bh = max(p.xyxy[2] - p.xyxy[0], 1.0), max(p.xyxy[3] - p.xyxy[1], 1.0)
            if not self._standing(st.zone, bw, bh):
                if st.sit_since is None:
                    st.sit_since = ts
                st.stand_since = None
            else:
                if st.stand_since is None:
                    st.stand_since = ts
                if ts - st.stand_since > cfg.sit_flicker_s:
                    st.sit_since = None
            horizon = max(self.seat_min_s, 1.0)
            while st.positions and st.positions[0][0] < ts - horizon:
                st.positions.popleft()
            # --- форма
            if self.uniform is not None and p.torso_hist is not None:
                u = self.uniform.score(p.torso_hist)
                st.uniform = u if st.uniform is None else 0.8 * st.uniform + 0.2 * u
            # --- внешний вид (классификатор): среднее логитов за окно
            if p.appearance is not None and np.isfinite(p.appearance):
                q = float(np.clip(p.appearance, 1e-3, 1 - 1e-3))
                st.app.append((ts, float(np.log(q / (1 - q)))))
            while st.app and st.app[0][0] < ts - cfg.appearance_window_s:
                st.app.popleft()
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
        standing = self._standing(st.zone, w, h)
        dwell = ts - st.zone_since if st.zone is not None else 0.0
        guest_scores, staff_scores, reasons = [], [], []
        # 1) долго в зоне стола и почти не двигается -> гость
        need = self.seat_min_s if not standing else max(self.seat_min_s, cfg.standing_guest_min_s)
        # Сидит — значит сидит seat_min_s без перерыва (дрожание рамки до sit_flicker_s прощается), а не кадр-другой:
        # человек, нагнувшийся протереть стол (постановка 15.11, подготовка), гостем не становится.
        sitting_s = ts - st.sit_since if st.sit_since is not None else 0.0
        if not standing and sitting_s < self.seat_min_s:
            need = max(self.seat_min_s, cfg.standing_guest_min_s)
        if st.zone is not None and dwell >= need and len(st.positions) >= 3:
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
        # 4) внешний вид (W2): среднее p(staff) по треку
        app_p = self.appearance_p(st)
        if app_p is not None:
            hi, lo = cfg.appearance_staff_thr, cfg.appearance_guest_thr
            if app_p >= hi:
                staff_scores.append(0.55 + 0.35 * (app_p - hi) / max(1e-6, 1 - hi))
                reasons.append("look_staff")
            elif app_p <= lo:
                guest_scores.append(0.55 + 0.3 * (lo - app_p) / max(1e-6, lo))
                reasons.append("look_guest")
        g, s = _noisy_or(guest_scores), _noisy_or(staff_scores)
        if g >= 0.5 and s >= 0.5 and abs(g - s) < cfg.conflict_margin:
            # сигналы расходятся (например, сидит долго, но выглядит как официант) — не угадываем
            return UNKNOWN, 0.5, (*reasons, "conflict"), standing, dwell
        if s >= 0.5 and s >= g:
            return STAFF, s, tuple(reasons), standing, dwell
        if g >= 0.5:
            return GUEST, g, tuple(reasons), standing, dwell
        return UNKNOWN, 1.0 - max(g, s), tuple(reasons), standing, dwell

    def _standing(self, zone: str | None, w: float, h: float) -> bool:
        """«Стоит» по рамке. Пропорции (h/w ≥ standing_aspect) — слабый признак: с высокой камеры у сидящего видны
        ноги под столом и рамка «стоячая» (постановка 15.11: сидящий 2,15 при пороге 1,9). Когда у стола уже
        сидели подтверждённые гости, порог подстраивается под камеру: стоит — кто заметно выше сидевших там
        (`standing_height_ratio` × медиана их высоты)."""
        cfg = self.cfg
        if h / w < cfg.standing_aspect:
            return False
        ref = self.seated_height(zone)
        return ref is None or h >= cfg.standing_height_ratio * ref

    def seated_height(self, zone: str | None) -> float | None:
        q = self._seated_h.get(zone) if zone else None
        if q is None or len(q) < max(1, self.cfg.seated_learn_min_samples):
            return None
        return float(np.median(q))

    def _learn_seated(self, st: _TrackState, reasons: tuple[str, ...]) -> None:
        """Образец «сидящий у этого стола»: подтверждённый гость, неподвижный в зоне (`dwell`), с рамкой не выше
        `seated_learn_max_aspect` (стоящий у барной стойки гость эталоном сидящего не станет)."""
        cfg = self.cfg
        if not cfg.seated_learn_min_samples or st.role != GUEST or "dwell" not in reasons or st.zone is None:
            return
        x1, y1, x2, y2 = st.xyxy
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        if h / w > cfg.seated_learn_max_aspect:
            return
        self._seated_h.setdefault(st.zone, deque(maxlen=600)).append(h)

    def appearance_p(self, st: _TrackState) -> float | None:
        if len(st.app) < max(1, self.cfg.appearance_min_samples):
            return None
        m = float(np.mean([v for _, v in st.app]))
        return 1.0 / (1.0 + np.exp(-m))

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
        self._learn_seated(st, reasons)
        return RoleDecision(st.role, round(st.confidence, 2), standing, round(dwell, 1), reasons, st.stable_id)

    def tables_visited(self, track_id: int, ts: float) -> int:
        st = self._tracks.get(track_id)
        if st is None:
            return 0
        return sum(1 for t in st.tables.values() if ts - t <= self.cfg.multi_table_window_s)
