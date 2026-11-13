"""W6: калибровка зала — обучение по меткам, версии, promote/rollback, страница проверки."""
import json

import numpy as np
import pytest

from qonaq.calib import store
from qonaq.calib.collect import auto_label, blurred_thumb
from qonaq.calib.train import format_report, train_site


@pytest.fixture()
def site(tmp_path):
    """Зал: 12 треков персонала (признак 0 ≈ +2) и 12 гостей (≈ −2), по 5 кропов."""
    rng = np.random.default_rng(0)
    X, track, tracks, labels = [], [], [], {}
    for tid in range(24):
        is_staff = tid < 12
        for _ in range(5):
            x = rng.normal(0, 1, 16)
            x[0] += 2 if is_staff else -2
            X.append(x)
            track.append(tid)
        auto = None if tid % 3 == 0 else ("staff" if is_staff else "guest")
        tracks.append({"key": f"t{tid}", "track_id": tid, "auto": auto, "thumbs": [], "seated_dwell_s": 0,
                       "max_tables": 0, "standing_share": 1.0 if is_staff else 0.0})
        if tid % 2 == 0:
            labels[f"t{tid}"] = "staff" if is_staff else "guest"
    d = store.calib_dir("cafe", tmp_path)
    d.mkdir(parents=True)
    np.savez(d / "samples.npz", X=np.array(X, np.float32), track=np.array(track), ts=np.zeros(len(X)),
             w=np.zeros(len(X)), h=np.zeros(len(X)), encoder=np.array("mnv2"))
    (d / "tracks.json").write_text(json.dumps(tracks))
    (d / "labels.json").write_text(json.dumps(labels))
    return tmp_path


def test_train_promote_rollback(site):
    r = train_site("cafe", root=site)
    assert r["version"] == 1 and r["n_test_tracks"] >= 2 and r["after"]["balanced_acc"] >= 0.85
    assert "было" in format_report(r)
    assert not store.active_path("cafe", site).exists()  # без promote голова не включается
    assert store.promote("cafe", root=site) == 1 and store.active_path("cafe", site).exists()
    r2 = train_site("cafe", root=site)
    assert r2["version"] == 2
    assert store.promote("cafe", 2, root=site) == 2 and store.active_version("cafe", site) == 2
    assert store.rollback("cafe", root=site) == 1 and store.active_version("cafe", site) == 1
    assert store.versions("cafe", site) == [1, 2]


def test_rollback_single_version_disables(site):
    train_site("cafe", root=site)
    store.promote("cafe", root=site)
    assert store.rollback("cafe", root=site) is None
    assert not store.active_path("cafe", site).exists()


def test_auto_label_rules():
    assert auto_label({"seated_dwell_s": 400, "standing_share": 0.1, "max_tables": 1}, 300, 3) == "guest"
    assert auto_label({"seated_dwell_s": 10, "standing_share": 0.9, "max_tables": 3}, 300, 3) == "staff"
    assert auto_label({"seated_dwell_s": 100, "standing_share": 0.5, "max_tables": 1}, 300, 3) is None


def test_thumb_is_blurred_on_top():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    th = blurred_thumb(frame, (100, 50, 160, 250))
    assert th is not None and th.shape[0] == 160
    top, bottom = th[:30].astype(float), th[-30:].astype(float)
    assert np.abs(np.diff(top, axis=1)).mean() < np.abs(np.diff(bottom, axis=1)).mean()  # верх пикселизован


def test_review_app(site):
    from fastapi.testclient import TestClient

    from qonaq.calib.review import create_app

    c = TestClient(create_app("cafe", site))
    assert "Калибровка" in c.get("/").text
    assert c.get("/api/tracks").json()["tracks"] == []  # нет кропов-картинок — нечего показывать
    assert c.post("/api/label", json={"key": "t1", "label": "staff"}).json()["ok"]
    assert store.load_labels("cafe", site)["t1"] == "staff"
    assert c.post("/api/label", json={"key": "t1", "label": "boss"}).status_code == 400
    assert c.get("/crops/../labels.json").status_code == 404
