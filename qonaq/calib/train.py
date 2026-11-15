"""`calibrate train` и кривая «сколько размеченных примеров зала → какая точность» (W6)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from qonaq.calib.store import active_path, load_labels, load_samples, load_tracks, new_version_dir
from qonaq.roles.head import Head, train_head
from qonaq.train.roles_eval import CLASSES, ROLE_C, binary_metrics, cap_train

log = logging.getLogger("qonaq.calib")

SITE_WEIGHT_MANUAL = 5.0  # вес вручную размеченного кропа зала относительно кропа общих датасетов
SITE_WEIGHT_AUTO = 2.0  # вес кропа с поведенческой авторазметкой


def general_features(encoder: str = "mnv2", feat_dir: Path = Path("data") / "features") -> tuple | None:
    """(X, y, venue) из кэша W2: эмбеддинг + HSV. None, если датасетов/кэша нет."""
    cands = sorted(feat_dir.glob(f"roles_{encoder}_*.npz"))
    if not cands:
        return None
    z = np.load(cands[-1], allow_pickle=False)
    ok = z["ok"].astype(bool)
    X = np.concatenate([z["emb"], z["hsv"]], axis=1)[ok]
    y = (z["label"] == "staff").astype(int)[ok]
    return X, y, z["venue"][ok]


def _split_tracks(keys: list[str], labels: dict[str, str], test_share: float, seed: int) -> tuple[set, set]:
    rng = np.random.default_rng(seed)
    test, train = set(), set()
    for lab in ("staff", "guest"):
        ks = [k for k in keys if labels.get(k) == lab]
        rng.shuffle(ks)
        n_test = int(round(len(ks) * test_share))
        if len(ks) >= 2:
            n_test = max(1, n_test)
        test |= set(ks[:n_test])
        train |= set(ks[n_test:])
    return train, test


def _split_time(ts: np.ndarray, keys: list[str], label_of, test_share: float) -> float | None:
    """Граница по времени: последние test_share размеченных кропов записи — проверка, всё раньше — обучение.
    None — времени у кропов нет (старые сборы) или в обучение не попадают оба класса."""
    lab = [(float(t), label_of(k)) for t, k in zip(ts, keys, strict=True) if label_of(k) in ("staff", "guest")]
    if len(lab) < 4 or max(t for t, _ in lab) - min(t for t, _ in lab) <= 0:
        return None
    times = np.array(sorted(t for t, _ in lab))
    t_split = float(times[min(len(times) - 1, int(round(len(times) * (1 - test_share))))])
    if len({c for t, c in lab if t < t_split}) < 2:
        return None
    return t_split


def train_site(site_id: str, general_head: Path | None = None, test_share: float = 0.3, seed: int = 0,
               root: Path | None = None, use_general_data: bool = False, split: str = "time") -> dict:
    """По умолчанию — только кропы зала: на отложенных залах W2 общие датасеты ухудшали результат
    (кривая в EVAL.md §8, D-022). use_general_data=True подмешивает их с весом 1 против 5/2 у зала.

    split="time" (по умолчанию, D-043): обучение — кропы до границы по времени, проверка — вручную размеченные
    кропы после неё. Раньше (split="tracks") в проверку уходили случайные треки, а одни и те же люди ходят по
    залу всю запись — соседние по времени треки того же официанта попадали и в обучение, и в проверку, и цифра
    «было → стало» получалась завышенной (постановка 15.11: EVAL §25)."""
    S = load_samples(site_id, root)
    tracks = {t["key"]: t for t in load_tracks(site_id, root)}
    manual = {k: v for k, v in load_labels(site_id, root).items() if v in ("staff", "guest")}
    keys = [f"t{int(t)}" for t in S["track"]]
    ts = np.asarray(S.get("ts", np.zeros(len(keys))), dtype=float)

    def label_of(k: str) -> str | None:
        return manual.get(k) or (tracks.get(k) or {}).get("auto")

    t_split = _split_time(ts, keys, label_of, test_share) if split == "time" else None
    if t_split is not None:
        is_test = np.array([t >= t_split for t in ts])
        test_mask = np.array([bool(m) and k in manual for k, m in zip(keys, is_test, strict=True)])
        test_keys = {k for k, m in zip(keys, test_mask, strict=True) if m}
    else:
        train_keys, test_keys = _split_tracks(sorted(manual), manual, test_share, seed)
        is_test = np.array([k in test_keys for k in keys])
        test_mask = is_test
    Xs, ys, ws = [], [], []
    for i, k in enumerate(keys):
        if is_test[i]:
            continue
        lab = manual.get(k) or (tracks.get(k) or {}).get("auto")
        if lab not in ("staff", "guest"):
            continue
        Xs.append(S["X"][i])
        ys.append(int(lab == "staff"))
        ws.append(SITE_WEIGHT_MANUAL if k in manual else SITE_WEIGHT_AUTO)
    if not Xs or len(set(ys)) < 2:
        raise SystemExit("мало размеченных кропов зала (нужны и персонал, и гости): "
                         "python -m qonaq calibrate review --site-id " + site_id)
    Xs, ys, ws = np.stack(Xs), np.array(ys), np.array(ws, float)
    gen = general_features(str(S["encoder"])) if use_general_data else None
    if gen is not None:
        Xg, yg, vg = gen
        sel = cap_train(np.arange(len(yg)), yg, vg, cap=3000)
        X = np.concatenate([Xg[sel], Xs])
        y = np.concatenate([yg[sel], ys])
        w = np.concatenate([np.ones(len(sel)), ws])
    else:
        log.warning("кэша общих датасетов нет — голова учится только на кропах зала")
        X, y, w = Xs, ys, ws
    head = train_head(X, y, CLASSES, C=ROLE_C, sample_weight=w, meta={
        "task": "roles", "features": f"{S['encoder']}+hsv", "encoder": str(S["encoder"]), "hsv": True,
        "site": site_id, "n_site": int(len(ys)), "n_general": int(len(y) - len(ys)),
        "created": time.strftime("%Y-%m-%d %H:%M")})
    # проверка на отложенных вручную размеченных кропах (после границы по времени или отложенных треках)
    te = test_mask
    report: dict = {"site": site_id, "n_manual_tracks": len(manual), "n_test_tracks": len(test_keys),
                    "n_test_crops": int(te.sum()), "train_crops_site": int(len(ys)),
                    "train_crops_general": int(len(y) - len(ys)),
                    "split": "time" if t_split is not None else "tracks",
                    "t_split_s": round(t_split, 1) if t_split is not None else None}
    if split == "time" and t_split is None:
        report["split_note"] = "у кропов нет времени или до границы один класс — проверка на отложенных треках"
    if te.any():
        yt = np.array([int(manual[k] == "staff") for k, m in zip(keys, te, strict=True) if m])
        new = binary_metrics(yt, head.predict_proba(S["X"][te])[:, 1])
        report["after"] = new
        base_path = general_head or (active_path(site_id, root) if active_path(site_id, root).exists() else None)
        if base_path is None:
            from qonaq.config import PROJECT_ROOT

            base_path = (root or PROJECT_ROOT) / "models" / "roles_head.npz"
        if Path(base_path).exists():
            old = Head.load(base_path)
            if old.in_dim == S["X"].shape[1]:
                report["before"] = binary_metrics(yt, old.predict_proba(S["X"][te])[:, 1])
                report["before_head"] = str(base_path)
    else:
        report["note"] = "нет отложенных ручных меток — качество не проверено"
    vdir = new_version_dir(site_id, root)
    head.save(vdir / "roles_head.npz")
    report["version"] = int(vdir.name[1:])
    (vdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def format_report(r: dict) -> str:
    def pct(m):
        return f"{100 * m['balanced_acc']:.1f}%" if m else "—"

    s = f"Версия v{r['version']}: кропов зала в обучении {r['train_crops_site']}, общих {r['train_crops_general']}."
    if r.get("split") == "time":
        s += f"\nРазделение по времени: обучение — до {r['t_split_s']:.0f} с записи, проверка — после."
    else:
        s += ("\nРазделение по трекам (случайно): соседние по времени кропы тех же людей — и в обучении, "
              "и в проверке; цифра завышена." + (f" ({r['split_note']})" if r.get("split_note") else ""))
    if "after" in r:
        s += (f"\nПроверка на {r['n_test_tracks']} отложенных треках ({r['n_test_crops']} кропов), balanced accuracy: "
              f"было {pct(r.get('before'))} → стало {pct(r['after'])}.")
        s += f"\nВключить: python -m qonaq calibrate promote --site-id {r['site']} --version {r['version']}"
    else:
        s += "\n" + r.get("note", "")
    return s


# ------------------------------------------------------------------ кривая обучения на отложенных залах (W6)
def calibration_curve(feats: dict, venues=("A", "B", "C"), ns=(0, 25, 50, 100, 300), seeds=(0, 1, 2),
                      weight: float = SITE_WEIGHT_MANUAL, split: str = "time") -> dict:
    """Для каждого отложенного зала: общая голова (другие залы) + n размеченных кропов этого зала.
    n кропов набираются поровну персонал/гость (как на странице проверки), если персонала хватает.

    split="time" (по умолчанию, D-043): калибровка — первая половина блоков времени зала, один блок — зазор,
    тест — остальное (как в жизни: размечаем запись при установке, работаем дальше). split="blocks" — старый
    способ: случайная половина блоков; половины чередуются внутри ролика, соседние кадры тех же людей попадают
    в обе — цифры завышены (EVAL §8, поправки B8 и §25)."""
    base = next(iter(feats.values()))
    ok = base["ok"].astype(bool)
    X = np.concatenate([base["emb"], base["hsv"]], axis=1)[ok]
    y = (base["label"] == "staff").astype(int)[ok]
    v = base["venue"][ok]
    g = base["group"][ok]
    out: dict = {}
    for venue in venues:
        res = {n: [] for n in ns}
        res_site_only = {n: [] for n in ns if n > 0}
        blocks = np.unique(g[v == venue])
        for seed in seeds:
            rng = np.random.default_rng(seed)
            if split == "time":
                order = sorted(blocks)
                half = len(order) // 2
                cal_blocks, test_blocks = set(order[:half]), set(order[half + 1:])  # блок order[half] — зазор
            else:
                perm = rng.permutation(blocks)
                cal_blocks = set(perm[: len(perm) // 2])
                test_blocks = set(blocks) - cal_blocks
            cal_pool = np.where((v == venue) & np.isin(g, list(cal_blocks)))[0]
            test = np.where((v == venue) & np.isin(g, list(test_blocks)))[0]
            if (y[test] == 1).sum() == 0 or (y[test] == 0).sum() == 0:
                continue
            gen = cap_train(np.where(v != venue)[0], y, v)
            for n in ns:
                if n == 0:
                    pick = np.array([], int)
                else:
                    st, gu = cal_pool[y[cal_pool] == 1], cal_pool[y[cal_pool] == 0]
                    k_s = min(len(st), n // 2)
                    pick = np.concatenate([rng.choice(st, k_s, replace=False),
                                           rng.choice(gu, min(len(gu), n - k_s), replace=False)])
                tr = np.concatenate([gen, pick])
                w = np.concatenate([np.ones(len(gen)), np.full(len(pick), weight)])
                head = train_head(X[tr], y[tr], CLASSES, C=ROLE_C, sample_weight=w)
                m = binary_metrics(y[test], head.predict_proba(X[test])[:, 1])
                res[n].append(m["balanced_acc"])
                if n > 0 and len(set(y[pick])) == 2:
                    h2 = train_head(X[pick], y[pick], CLASSES, C=ROLE_C)
                    res_site_only[n].append(binary_metrics(y[test], h2.predict_proba(X[test])[:, 1])["balanced_acc"])
            log.info("кривая %s seed=%d: %s", venue, seed, {n: round(r[-1], 3) for n, r in res.items() if r})
        out[venue] = {
            "general_plus_site": {n: {"mean": float(np.mean(r)), "min": float(np.min(r)), "max": float(np.max(r)),
                                      "runs": len(r)} for n, r in res.items() if r},
            "site_only": {n: {"mean": float(np.mean(r)), "runs": len(r)} for n, r in res_site_only.items() if r},
            "test_crops_last_seed": int(len(test)),
        }
    return out


def curve_markdown(c: dict) -> str:
    ns = sorted({int(n) for r in c.values() for n in r["general_plus_site"]})
    L = ["| Зал | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
    for venue, r in c.items():
        cells = []
        for n in ns:
            m = r["general_plus_site"].get(n) or r["general_plus_site"].get(str(n))
            cells.append(f"{m['mean']:.3f} ({m['min']:.2f}–{m['max']:.2f})" if m else "—")
        L.append(f"| {venue} | " + " | ".join(cells) + " |")
    pos = [n for n in ns if n > 0]
    L += ["", "Только кропы зала, без общих датасетов:", "",
          "| Зал | " + " | ".join(f"n={n}" for n in pos) + " |", "|---|" + "---|" * len(pos)]
    for venue, r in c.items():
        cells = []
        for n in ns:
            if n == 0:
                continue
            m = r["site_only"].get(n) or r["site_only"].get(str(n))
            cells.append(f"{m['mean']:.3f}" if m else "—")
        L.append(f"| {venue} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"
