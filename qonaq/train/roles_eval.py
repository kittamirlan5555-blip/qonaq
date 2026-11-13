"""W2: оценка классификатора «персонал / гость» leave-one-venue-out и обучение итоговой головы.

    python -m qonaq train roles                      # оценка всех вариантов + итоговая голова
    python -m qonaq train roles --encoders mnv2      # только один энкодер

Результаты: runs/train/roles_eval.json, runs/train/roles_eval.md; голова — models/roles_head.npz.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from qonaq.datasets.registry import MAIN_VENUES
from qonaq.datasets.splits import group_kfold, leave_one_venue_out
from qonaq.roles.head import Head, train_head
from qonaq.train.featcache import extract

log = logging.getLogger("qonaq.train")
CLASSES = ["guest", "staff"]
PR_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    """y: 1 = staff. Матрица ошибок [[гость→гость, гость→персонал], [персонал→гость, персонал→персонал]]."""
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    rec_s = tp / (tp + fn) if tp + fn else float("nan")
    rec_g = tn / (tn + fp) if tn + fp else float("nan")
    prec_s = tp / (tp + fp) if tp + fp else float("nan")
    bal = np.nanmean([rec_s, rec_g])
    return {"n": int(len(y)), "n_staff": int((y == 1).sum()), "n_guest": int((y == 0).sum()),
            "balanced_acc": float(bal), "staff_precision": float(prec_s), "staff_recall": float(rec_s),
            "guest_recall": float(rec_g), "confusion": [[tn, fp], [fn, tp]]}


def pr_points(y: np.ndarray, p: np.ndarray) -> list[dict]:
    pts = []
    for t in PR_THRESHOLDS:
        m = binary_metrics(y, p, t)
        pts.append({"thr": t, "precision": m["staff_precision"], "recall": m["staff_recall"]})
    return pts


def build_X(feats: dict[str, dict], cfg: str) -> np.ndarray:
    """cfg: 'hsv' | '<enc>' | '<enc>+hsv'."""
    parts = []
    for p in cfg.split("+"):
        if p == "hsv":
            parts.append(next(iter(feats.values()))["hsv"])
        else:
            parts.append(feats[p]["emb"])
    return np.concatenate(parts, axis=1)


ROLE_C = 0.01  # сильная регуляризация: при перекрёстных залах чуть лучше (перебор C — EVAL.md §6)
TRAIN_CAP = 3000  # не больше стольких кропов одного класса из одного зала в обучении (скорость; тест — полный)


def cap_train(tr: np.ndarray, y: np.ndarray, venues: np.ndarray, cap: int = TRAIN_CAP, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for v in np.unique(venues[tr]):
        for c in (0, 1):
            idx = tr[(venues[tr] == v) & (y[tr] == c)]
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            keep.append(idx)
    return np.sort(np.concatenate(keep))


def _eval_split(X, y, tr, te, small, kind="logreg", venues=None) -> tuple[dict, Head, np.ndarray]:
    if venues is not None:
        tr = cap_train(tr, y, venues)
    head = train_head(X[tr], y[tr], CLASSES, kind=kind, C=ROLE_C)
    p = head.predict_proba(X[te])[:, 1]
    yt = y[te]
    st = small[te] == 1
    res = {"all": binary_metrics(yt, p), "normal": binary_metrics(yt[~st], p[~st]),
           "small": binary_metrics(yt[st], p[st]) if st.any() else None, "pr": pr_points(yt, p)}
    return res, head, p


def evaluate(feats: dict[str, dict], cfgs: list[str], kinds: tuple[str, ...] = ("logreg",)) -> dict:
    base = next(iter(feats.values()))
    ok = base["ok"].astype(bool)
    for f in feats.values():
        ok &= f["ok"].astype(bool)
    y_all = (base["label"] == "staff").astype(int)
    venues = base["venue"]
    small = base["small"]
    out: dict = {}
    for cfg in cfgs:
        X_all = build_X(feats, cfg)
        for kind in kinds:
            key = f"{cfg}|{kind}"
            X, y, v, sm, grp = X_all[ok], y_all[ok], venues[ok], small[ok], base["group"][ok]
            per = {}
            for venue, tr, te in leave_one_venue_out(v):
                t0 = time.time()
                res, _, _ = _eval_split(X, y, tr, te, sm, kind, venues=v)
                res["train_s"] = round(time.time() - t0, 1)
                per[venue] = res
                log.info("%s held=%s bal=%.3f P=%.3f R=%.3f", key, venue, res["all"]["balanced_acc"],
                         res["all"]["staff_precision"], res["all"]["staff_recall"])
            main = [per[x]["all"] for x in MAIN_VENUES if x in per]
            summary = {k: float(np.nanmean([m[k] for m in main]))
                       for k in ("balanced_acc", "staff_precision", "staff_recall")}
            # «верхняя граница»: тот же зал в обучении (другие блоки времени), группы не пересекаются
            upper = {}
            for venue in MAIN_VENUES:
                idx_v = np.where(v == venue)[0]
                if len(idx_v) == 0:
                    continue
                ps, ys = [], []
                for _f, tr_v, te_v in group_kfold(grp[idx_v], k=5):
                    tr = np.concatenate([np.where(v != venue)[0], idx_v[tr_v]])
                    te = idx_v[te_v]
                    if (y[te] == 1).sum() == 0:
                        continue
                    head = train_head(X[cap_train(tr, y, v)], y[cap_train(tr, y, v)], CLASSES, kind=kind, C=ROLE_C)
                    ps.append(head.predict_proba(X[te])[:, 1])
                    ys.append(y[te])
                if ps:
                    upper[venue] = binary_metrics(np.concatenate(ys), np.concatenate(ps))
            out[key] = {"per_venue": per, "mean_ABC": summary, "upper_bound_within_venue": upper,
                        "upper_mean_ABC": float(np.nanmean([u["balanced_acc"] for u in upper.values()]))
                        if upper else None}
    return out


def to_markdown(res: dict, speed: dict) -> str:
    L = ["# W2 — персонал / гость: leave-one-venue-out", "",
         "Голова обучается на трёх залах, проверяется на четвёртом. Порог p(staff) ≥ 0.5. "
         "Главный показатель — среднее по A, B, C. «Верхняя граница» — 5 фолдов по блокам времени внутри зала "
         "(обучение = другие залы + другие блоки этого зала).", "",
         "| Вариант | bal.acc A | bal.acc B | bal.acc C | bal.acc D | **среднее A,B,C** | P(staff) ср. | R(staff) ср. | верхняя граница ср. |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, r in res.items():
        pv = r["per_venue"]
        cells = [f"{pv[v]['all']['balanced_acc']:.3f}" if v in pv else "—" for v in ("A", "B", "C", "D")]
        m = r["mean_ABC"]
        ub = r["upper_mean_ABC"]
        ub_s = f"{ub:.3f}" if ub is not None else "—"
        L.append(f"| {key} | {' | '.join(cells)} | **{m['balanced_acc']:.3f}** | {m['staff_precision']:.3f} | "
                 f"{m['staff_recall']:.3f} | {ub_s} |")
    L += ["", "## Детали по залам (лучший вариант по среднему A,B,C)", ""]
    best = max(res, key=lambda k: res[k]["mean_ABC"]["balanced_acc"])
    L.append(f"Вариант: `{best}`")
    L.append("")
    L.append("| Зал | кропов (гость/персонал) | bal.acc | P(staff) | R(staff) | матрица [[г→г, г→п],[п→г, п→п]] | "
             "мелкие: n, bal.acc | PR-точки (порог: P/R) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for v, r in res[best]["per_venue"].items():
        a = r["all"]
        s = r["small"]
        sm = f"{s['n']}, {s['balanced_acc']:.3f}" if s else "0"
        pr = "; ".join(f"{p['thr']}: {p['precision']:.2f}/{p['recall']:.2f}" for p in r["pr"][::2])
        L.append(f"| {v} | {a['n_guest']}/{a['n_staff']} | {a['balanced_acc']:.3f} | {a['staff_precision']:.3f} | "
                 f"{a['staff_recall']:.3f} | {a['confusion']} | {sm} | {pr} |")
    if speed:
        L += ["", "## Скорость энкодеров (CPU этой машины, батч 32)", "",
              "| Энкодер | вход | мс/кроп |", "|---|---|---|"]
        for k, s in speed.items():
            L.append(f"| {k} | {s['input_hw']} | {s['ms_per_crop']:.1f} |")
    return "\n".join(L) + "\n"


def bench(encoder: str, n: int = 128) -> dict:
    from qonaq.roles.encoder import OnnxEncoder

    e = OnnxEncoder(encoder)
    crops = [np.random.default_rng(i).integers(0, 255, (150, 60, 3), dtype=np.uint8) for i in range(n)]
    e.embed_crops(crops[:32])
    t = time.time()
    e.embed_crops(crops)
    return {"input_hw": list(e.input_hw), "ms_per_crop": (time.time() - t) / n * 1000}


def run(index_csv: Path, encoders: list[str], out_dir: Path, models_dir: Path, final: str | None = None) -> dict:
    feat_dir = Path("data") / "features"
    feats = {e: extract(index_csv, e, feat_dir) for e in encoders}
    cfgs = ["hsv"] + [e for e in encoders] + [f"{e}+hsv" for e in encoders]
    res = evaluate(feats, cfgs, kinds=("logreg",))
    best_cfg = max((k for k in res if k != "hsv|logreg"), key=lambda k: res[k]["mean_ABC"]["balanced_acc"])
    cfg_best = best_cfg.split("|")[0]
    res.update(evaluate(feats, [cfg_best], kinds=("mlp",)))
    speed = {}
    for e in encoders:
        speed[e] = bench(e)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "roles_eval.json").write_text(json.dumps({"results": res, "speed": speed}, ensure_ascii=False,
                                                        indent=1, default=float), encoding="utf-8")
    (out_dir / "roles_eval.md").write_text(to_markdown(res, speed), encoding="utf-8")
    # итоговая голова: выбранный вариант, все залы
    cfg_final = final or cfg_best
    kind = "logreg"
    base = feats[encoders[0]]
    ok = np.ones(len(base["label"]), bool)
    for f in feats.values():
        ok &= f["ok"].astype(bool)
    X = build_X(feats, cfg_final)[ok]
    y = (base["label"] == "staff").astype(int)[ok]
    sel = cap_train(np.arange(len(y)), y, base["venue"][ok], cap=6000)
    X, y = X[sel], y[sel]
    enc = [p for p in cfg_final.split("+") if p != "hsv"]
    head = train_head(X, y, CLASSES, kind=kind, C=ROLE_C, meta={
        "task": "roles", "features": cfg_final, "encoder": enc[0] if enc else None, "hsv": "hsv" in cfg_final,
        "trained_on": sorted(set(base["venue"][ok].tolist())), "n": int(len(y)),
        "created": time.strftime("%Y-%m-%d"), "eval_mean_ABC": res[f"{cfg_final}|{kind}"]["mean_ABC"]})
    path = head.save(models_dir / "roles_head.npz")
    log.info("итоговая голова: %s (%s)", path, cfg_final)
    return {"results": res, "speed": speed, "head": str(path), "final_cfg": cfg_final}
