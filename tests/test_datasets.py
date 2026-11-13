"""W1/W2: схема датасетов, дедупликация, честные разбиения, голова классификатора."""
from pathlib import Path

import numpy as np

from qonaq.datasets.index import is_small, role_group, table_camera
from qonaq.datasets.registry import DATASETS
from qonaq.datasets.splits import group_kfold, leave_one_venue_out
from qonaq.datasets.yolo import YoloImage, dedupe, parse_label_file, source_key
from qonaq.roles.head import Head, train_head


def test_source_key_strips_roboflow_suffix():
    assert source_key("cam1_1hour_frame_0001_jpg.rf.16ec0e94185c9f30dee451036f6a6ad5.jpg") == "cam1_1hour_frame_0001"
    assert source_key("21edd85f-81af_jpeg.rf.c42f.jpg") == "21edd85f-81af"
    assert source_key("plain_name.jpg") == "plain_name"


def test_dedupe_prefers_valid_over_train():
    ims = [YoloImage(Path("a.jpg"), "train", "s1", []), YoloImage(Path("b.jpg"), "valid", "s1", []),
           YoloImage(Path("c.jpg"), "train", "s2", [])]
    out = dedupe(ims)
    assert [i.source for i in out] == ["s1", "s2"]
    assert out[0].split == "valid"


def test_parse_label_file_box_and_polygon(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n9 0.5 0.5 0.1 0.1\n")
    b = parse_label_file(p, ["Customer", "Staff"])
    assert len(b) == 2  # класс 9 вне списка — пропуск
    assert b[0].cls_name == "Customer" and abs(b[0].x1 - 0.4) < 1e-6 and abs(b[0].y2 - 0.7) < 1e-6
    assert b[1].cls_name == "Staff" and abs(b[1].x2 - 0.3) < 1e-6


def test_unified_schema():
    assert DATASETS["smart_restaurant_a"].class_map.get("person") is None  # person игнорируется
    assert DATASETS["smart_restaurant_a"].class_map["waiter"] == "staff"
    assert DATASETS["worker_customer_d"].class_map["Worker"] == "staff"
    assert DATASETS["ramada_c"].class_map == {"Waiter": "staff"}
    assert DATASETS["clean_table"].class_map["unoccupied_dirty"] == "dirty"
    assert not DATASETS["table_detection"].train_ok


def test_groups_and_small():
    spec = DATASETS["smart_restaurant_a"]
    assert role_group(spec, "image_00499") == "A00" and role_group(spec, "image_00500") == "A01"
    assert role_group(DATASETS["worker_customer_d"], "1_IPC_main_20250107131126") == "D20250107"
    assert table_camera("cam5_1hour_frame_0002") == "cam5"
    assert table_camera("photo_100_2025-05-06_09-55-16") == "photo"
    assert is_small(20, 40) == 1 and is_small(40, 30) == 0 and is_small(100, 12) == 1


def test_leave_one_venue_out_disjoint():
    v = np.array(list("AAABBBCCD"))
    folds = list(leave_one_venue_out(v))
    assert [f[0] for f in folds] == ["A", "B", "C", "D"]
    for venue, tr, te in folds:
        assert set(v[te]) == {venue} and venue not in set(v[tr])


def test_group_kfold_keeps_groups_whole():
    g = np.array(["g1"] * 5 + ["g2"] * 3 + ["g3"] * 4 + ["g4"] * 2 + ["g5"] * 6)
    seen = set()
    for _f, tr, te in group_kfold(g, k=3):
        assert not set(g[tr]) & set(g[te])
        seen |= set(te.tolist())
    assert seen == set(range(len(g)))


def test_head_roundtrip_numpy_matches_sklearn(tmp_path):
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (200, 5)), rng.normal(2, 1, (40, 5))])
    y = np.array([0] * 200 + [1] * 40)
    for kind in ("logreg", "mlp"):
        h = train_head(X, y, ["guest", "staff"], kind=kind, meta={"task": "t"})
        p = h.predict_proba(X)
        assert p.shape == (240, 2) and np.allclose(p.sum(1), 1)
        bal = ((p[:200, 1] < 0.5).mean() + (p[200:, 1] >= 0.5).mean()) / 2
        assert bal > 0.85
        path = h.save(tmp_path / f"{kind}.npz")
        h2 = Head.load(path)
        assert np.allclose(h2.predict_proba(X), p, atol=1e-5)
        assert h2.meta["task"] == "t" and h2.classes == ["guest", "staff"]
