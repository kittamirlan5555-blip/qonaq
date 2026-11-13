"""Лёгкая голова классификатора поверх признаков (W2/W3, DECISIONS D-017).

Обучение — scikit-learn (только при обучении/калибровке); рантайм — чистый numpy из `.npz`
(стандартизация + логрегрессия или MLP с одним скрытым слоем). Файл весит десятки–сотни КБ.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Head:
    def __init__(self, classes: list[str], mean: np.ndarray, scale: np.ndarray,
                 layers: list[tuple[np.ndarray, np.ndarray]], meta: dict | None = None):
        self.classes = list(classes)
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.layers = [(w.astype(np.float32), b.astype(np.float32)) for w, b in layers]
        self.meta = meta or {}

    @property
    def in_dim(self) -> int:
        return int(self.mean.shape[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(N, n_classes). Бинарная голова — тоже 2 столбца."""
        z = (np.asarray(X, np.float32) - self.mean) / self.scale
        for i, (w, b) in enumerate(self.layers):
            z = z @ w + b
            if i < len(self.layers) - 1:
                z = np.maximum(z, 0)
        if z.shape[1] == 1:
            p = 1.0 / (1.0 + np.exp(-np.clip(z[:, 0], -50, 50)))
            return np.stack([1 - p, p], axis=1)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrs = {"mean": self.mean, "scale": self.scale}
        for i, (w, b) in enumerate(self.layers):
            arrs[f"w{i}"] = w
            arrs[f"b{i}"] = b
        meta = dict(self.meta, classes=self.classes, n_layers=len(self.layers))
        np.savez_compressed(path, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrs)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Head:
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(meta["n_layers"]))]
        return cls(meta["classes"], z["mean"], z["scale"], layers, meta)


def train_head(X: np.ndarray, y: np.ndarray, classes: list[str], kind: str = "logreg", C: float = 0.5,
               sample_weight: np.ndarray | None = None, seed: int = 0, meta: dict | None = None) -> Head:
    """y — индексы классов. kind: logreg | mlp. Классы взвешиваются обратно частоте (balanced)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    sc = StandardScaler().fit(X, sample_weight=sample_weight)
    Xs = sc.transform(X)
    counts = np.bincount(y, minlength=len(classes)).astype(float)
    cw = np.where(counts > 0, len(y) / (len(classes) * np.maximum(counts, 1)), 0.0)
    w = cw[y] * (sample_weight if sample_weight is not None else 1.0)
    scale = np.where(sc.scale_ > 0, sc.scale_, 1.0)
    if kind == "logreg":
        m = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        m.fit(Xs, y, sample_weight=w)
        coef, icpt = m.coef_.T, m.intercept_
        layers = [(coef, icpt)]
    elif kind == "mlp":
        # MLPClassifier не умеет sample_weight в старых версиях — балансируем повтором
        rng = np.random.default_rng(seed)
        rep = np.clip(np.round(cw[y]).astype(int), 1, 20)
        idx = np.repeat(np.arange(len(y)), rep)
        rng.shuffle(idx)
        m = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=200, early_stopping=True,
                          random_state=seed)
        m.fit(Xs[idx], y[idx])
        layers = list(zip(m.coefs_, m.intercepts_, strict=True))
    else:
        raise ValueError(kind)
    return Head(classes, sc.mean_, scale, layers, dict(meta or {}, kind=kind))
