"""W3: классификатор кропа стола clean / dirty / occupied — разбиение по камерам + внешний тест.

    python -m qonaq train tables
Результаты: runs/train/tables_eval.{json,md}; голова — models/tables_head.npz.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from qonaq.roles.head import train_head
from qonaq.train.featcache import extract

log = logging.getLogger("qonaq.train")
CLASSES = ["clean", "dirty", "occupied"]


def multiclass_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    k = len(CLASSES)
    cm = np.zeros((k, k), int)
    for a, b in zip(y, pred, strict=True):
        cm[a, b] += 1
    rec = {CLASSES[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i in range(k)}
    prec = {CLASSES[i]: (float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else None) for i in range(k)}
    present = [r for r in rec.values() if r is not None]
    return {"n": int(len(y)), "per_class_n": {CLASSES[i]: int(cm[i].sum()) for i in range(k)},
            "recall": rec, "precision": prec, "balanced_acc": float(np.mean(present)) if present else None,
            "confusion": cm.tolist()}


def clean_vs_dirty(y: np.ndarray, p: np.ndarray, thr: float = 0.6) -> dict:
    """Как сигнал используется в рантайме: только незанятые столы, r = p(dirty)/(p(dirty)+p(clean))."""
    m = y != CLASSES.index("occupied")
    if not m.any():
        return {}
    r = p[m, 1] / np.maximum(p[m, 0] + p[m, 1], 1e-6)
    yt = y[m]
    out = {"n_clean": int((yt == 0).sum()), "n_dirty": int((yt == 1).sum())}
    say_dirty, say_clean = r >= thr, r <= 1 - thr
    out["dirty_recall"] = float((say_dirty & (yt == 1)).sum() / max(1, (yt == 1).sum())) if (yt == 1).any() else None
    out["clean_recall"] = float((say_clean & (yt == 0)).sum() / max(1, (yt == 0).sum())) if (yt == 0).any() else None
    out["abstain"] = float((~say_dirty & ~say_clean).mean())
    # ошибочный «чистый» — самая дорогая ошибка (стол отпустили грязным)
    out["false_clean_of_dirty"] = float((say_clean & (yt == 1)).sum() / max(1, (yt == 1).sum())) if (yt == 1).any() else None
    return out


def build(feats: dict, cfg: str) -> np.ndarray:
    parts = [feats["hsv"]] if cfg == "hsv" else ([feats["emb"]] + ([feats["hsv"]] if cfg.endswith("+hsv") else []))
    return np.concatenate(parts, axis=1)


def evaluate(feats: dict, cfgs: list[str], C: float = 0.1) -> dict:
    lab = feats["label"]
    y = np.array([CLASSES.index(v) for v in lab])
    ok = feats["ok"].astype(bool)
    grp = np.array(["photo" if g in ("photo", "other") else g for g in feats["venue"]])
    ext = feats["dataset"] == "table_detection"
    res: dict = {}
    for cfg in cfgs:
        X = build(feats, cfg)
        per = {}
        train_pool = np.where(ok & ~ext)[0]
        for g in sorted(set(grp[train_pool])):
            te = train_pool[grp[train_pool] == g]
            tr = train_pool[grp[train_pool] != g]
            head = train_head(X[tr], y[tr], CLASSES, C=C)
            p = head.predict_proba(X[te])
            per[g] = {**multiclass_metrics(y[te], p.argmax(1)), "clean_vs_dirty": clean_vs_dirty(y[te], p)}
            log.info("tables %s held=%s bal=%.3f", cfg, g, per[g]["balanced_acc"] or -1)
        head = train_head(X[train_pool], y[train_pool], CLASSES, C=C)
        te = np.where(ok & ext)[0]
        p = head.predict_proba(X[te])
        external = {**multiclass_metrics(y[te], p.argmax(1)), "clean_vs_dirty": clean_vs_dirty(y[te], p)}
        res[cfg] = {"per_camera": per, "external": external}
    return res


def to_markdown(res: dict) -> str:
    L = ["# W3 — состояние стола: clean / dirty / occupied", "",
         "Разбиение по камерам: голова учится на всех группах, кроме одной, проверяется на ней "
         "(cam1, cam2, cam3, cam5, cam6 из Clean.Table; `photo` — все кадры `photo_*` из Dirty.Table). "
         "Внешний тест — Table.Detection (69 кадров, другие камеры), голова обучена на всех группах.", "",
         "| Вариант | внешний тест: bal.acc | recall clean / dirty / occupied | «грязный»→«чистый» на внешнем | "
         "photo (только грязные): recall dirty |", "|---|---|---|---|---|"]
    for cfg, r in res.items():
        e = r["external"]
        rc = " / ".join(f"{e['recall'][c]:.2f}" if e["recall"][c] is not None else "—" for c in CLASSES)
        fc = e["clean_vs_dirty"].get("false_clean_of_dirty")
        ph = r["per_camera"].get("photo", {}).get("recall", {}).get("dirty")
        L.append(f"| {cfg} | {e['balanced_acc']:.3f} | {rc} | {fc:.2f} | {ph:.2f} |" if fc is not None and ph is not None
                 else f"| {cfg} | {e['balanced_acc']:.3f} | {rc} | — | — |")
    best = max(res, key=lambda k: res[k]["external"]["balanced_acc"])
    L += ["", f"## По камерам, вариант `{best}`", "",
          "| Группа | n (clean/dirty/occ) | bal.acc | recall clean / dirty / occupied | матрица (строки — правда) |",
          "|---|---|---|---|---|"]
    for g, m in res[best]["per_camera"].items():
        n = m["per_class_n"]
        rc = " / ".join(f"{m['recall'][c]:.2f}" if m["recall"][c] is not None else "—" for c in CLASSES)
        L.append(f"| {g} | {n['clean']}/{n['dirty']}/{n['occupied']} | {m['balanced_acc']:.3f} | {rc} | {m['confusion']} |")
    e = res[best]["external"]
    L += ["", f"Внешний тест, `{best}`: матрица {e['confusion']} (строки/столбцы: {', '.join(CLASSES)}); "
          f"сигнал clean/dirty с порогом 0.6: {json.dumps(e['clean_vs_dirty'], ensure_ascii=False)}"]
    return "\n".join(L) + "\n"


def run(index_csv: Path, out_dir: Path, models_dir: Path, encoder: str = "mnv2") -> dict:
    feats = extract(index_csv, encoder, Path("data") / "features", kind="table")
    res = evaluate(feats, ["hsv", encoder, f"{encoder}+hsv"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "tables_eval.md").write_text(to_markdown(res), encoding="utf-8")
    best = max(res, key=lambda k: res[k]["external"]["balanced_acc"])
    lab = feats["label"]
    y = np.array([CLASSES.index(v) for v in lab])
    m = feats["ok"].astype(bool)  # итоговая голова: все данные, включая внешний тест
    X = build(feats, best)
    enc = None if best == "hsv" else encoder
    head = train_head(X[m], y[m], CLASSES, C=0.1, meta={
        "task": "tables", "features": best, "encoder": enc, "hsv": best == "hsv" or best.endswith("+hsv"),
        "n": int(m.sum()), "created": time.strftime("%Y-%m-%d"),
        "eval_external_bal_acc": res[best]["external"]["balanced_acc"]})
    path = head.save(models_dir / "tables_head.npz")
    return {"results": res, "best": best, "head": str(path)}


def dish_eval(index_csv: Path, out_dir: Path, model: str = "yolox_m", per_group: int = 150, seed: int = 0) -> dict:
    """Сигнал «посуда в зоне стола» без обучения: детектор COCO (cup, bowl, bottle, …) внутри bbox стола,
    люди не касаются bbox. Считается на подвыборке кадров каждой группы (камеры/photo/ext)."""
    import cv2

    from qonaq.datasets.index import load_rows
    from qonaq.detect.base import DISH_LABELS, PERSON, class_ids
    from qonaq.detect.onnx_yolox import OnnxYoloxDetector

    rows = load_rows(index_csv)
    by_img: dict[str, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for img, rs in by_img.items():
        g = rs[0]["venue"]
        g = "photo" if g in ("photo", "other") else ("cam" if g.startswith("cam") else g)
        groups.setdefault(g, []).append(img)
    det = OnnxYoloxDetector(model=model, conf=0.2, classes=class_ids(True))
    res: dict = {}
    for g, imgs in groups.items():
        pick = imgs if len(imgs) <= per_group else list(rng.choice(imgs, per_group, replace=False))
        stats: dict = {}
        for img in pick:
            frame = cv2.imread(img)
            if frame is None:
                continue
            dets = det.detect([frame])[0]
            people = [d for d in dets if d.label == PERSON and d.conf >= 0.3]
            for r in by_img[img]:
                x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
                hidden = any(p.x1 < x2 and p.x2 > x1 and p.y1 < y2 and p.y2 > y1 for p in people)
                n = sum(1 for d in dets if d.label in DISH_LABELS and d.conf >= 0.3
                        and x1 <= (d.x1 + d.x2) / 2 <= x2 and y1 <= (d.y1 + d.y2) / 2 <= y2)
                s = stats.setdefault(r["label"], {"n": 0, "hidden": 0, "with_dishes": 0})
                s["n"] += 1
                if hidden:
                    s["hidden"] += 1
                elif n >= 1:
                    s["with_dishes"] += 1
        for s in stats.values():
            vis = s["n"] - s["hidden"]
            s["share_with_dishes_of_visible"] = round(s["with_dishes"] / vis, 3) if vis else None
        res[g] = {"images": len(pick), "by_label": stats}
        log.info("dishes %s: %s", g, stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables_dishes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res
