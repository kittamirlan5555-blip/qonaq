"""M2: модели зон, YAML, привязка людей к зонам, автопредложение, логика редактора."""
import numpy as np
import pytest

from qonaq.detect.base import Detection
from qonaq.zones.assign import ZoneAssigner, bbox_overlap_fraction, point_in_polygon
from qonaq.zones.editor import EditorState
from qonaq.zones.models import Zone, ZoneSet, load_zones, save_zones
from qonaq.zones.suggest import cluster_boxes, expand_table, reading_order, suggest_from_detections

SQ = [(100, 100), (300, 100), (300, 300), (100, 300)]


def zs(*zones) -> ZoneSet:
    return ZoneSet(camera_id="cam", frame_size=(640, 480), zones=list(zones))


def test_zone_yaml_roundtrip_all_types(tmp_path):
    src = zs(
        Zone(id="T01", type="table", name="У окна", polygon=SQ, seats=4),
        Zone(id="E01", type="entrance", polygon=[(0, 0), (50, 0), (50, 50)]),
        Zone(id="W01", type="waiting_area", polygon=SQ),
        Zone(id="S01", type="service_point", polygon=SQ, staff_polygon=[(0, 0), (9, 0), (9, 9)]),
        Zone(id="X01", type="ignore", polygon=SQ),
    )
    back = load_zones(save_zones(src, tmp_path / "z.yaml"))
    assert [z.type for z in back.zones] == ["table", "entrance", "waiting_area", "service_point", "ignore"]
    assert back.zones[0].name == "У окна" and back.zones[0].seats == 4
    assert back.frame_size == (640, 480)
    assert back.zones[3].staff_polygon[1] == (9.0, 0.0)


def test_zone_validation():
    with pytest.raises(ValueError):
        Zone(id="T01", polygon=[(0, 0), (1, 1)])
    with pytest.raises(ValueError):
        zs(Zone(id="T01", polygon=SQ), Zone(id="T01", polygon=SQ))
    with pytest.raises(ValueError):
        Zone(id="T01", type="sofa", polygon=SQ)


def test_default_anchor_by_type():
    assert Zone(id="T01", type="table", polygon=SQ).anchor_strategy == "overlap"
    assert Zone(id="E01", type="entrance", polygon=SQ).anchor_strategy == "bottom_center"
    assert Zone(id="T02", type="table", polygon=SQ, anchor="center").anchor_strategy == "center"


def test_zoneset_scaling():
    s = zs(Zone(id="T01", polygon=SQ)).scaled(1280, 960)
    assert s.zones[0].polygon[0] == (200.0, 200.0) and s.frame_size == (1280, 960)


def test_point_in_polygon_concave():
    poly = np.array([(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)], np.float32)  # «галочка»
    assert point_in_polygon((2, 2), poly)
    assert not point_in_polygon((5, 8), poly)


def test_bbox_overlap_fraction():
    poly = np.array(SQ, np.float32)
    assert bbox_overlap_fraction((150, 150, 250, 250), poly) == pytest.approx(1.0)
    assert bbox_overlap_fraction((200, 100, 400, 300), poly) == pytest.approx(0.5, abs=0.08)
    assert bbox_overlap_fraction((400, 400, 500, 500), poly) == 0.0


def test_bottom_center_assignment():
    a = ZoneAssigner(zs(Zone(id="E01", type="entrance", polygon=SQ)))
    assert a.assign_one((150, 50, 250, 200)) == ["E01"]  # ноги внутри
    assert a.assign_one((150, 150, 250, 400)) == []  # ноги ниже зоны, хотя тело внутри


def test_overlap_assignment_for_seated_person_with_hidden_legs():
    """Сидящий: bbox обрезан столом, нижняя точка вне зоны — overlap всё равно сажает его за стол."""
    a = ZoneAssigner(zs(Zone(id="T01", type="table", polygon=SQ)))
    assert a.assign_one((120, 60, 220, 200)) == ["T01"]


def test_person_goes_to_table_with_larger_overlap():
    t1 = Zone(id="T01", polygon=[(0, 0), (200, 0), (200, 200), (0, 200)])
    t2 = Zone(id="T02", polygon=[(150, 0), (400, 0), (400, 200), (150, 200)])
    a = ZoneAssigner(zs(t1, t2))
    assert a.assign_one((140, 20, 240, 180)) == ["T02"]
    assert a.assign_one((60, 20, 180, 180)) == ["T01"]


def test_ignore_zone_overrides_everything():
    t = Zone(id="T01", polygon=SQ)
    x = Zone(id="X01", type="ignore", polygon=[(0, 250), (640, 250), (640, 480), (0, 480)])
    a = ZoneAssigner(zs(t, x))
    assert a.assign_one((150, 120, 250, 280)) == []  # ноги в ignore
    assert a.assign_one((150, 100, 250, 240)) == ["T01"]


def test_table_and_passage_zone_together():
    t = Zone(id="T01", polygon=SQ)
    e = Zone(id="E01", type="entrance", polygon=[(0, 0), (640, 0), (640, 480), (0, 480)])
    assert ZoneAssigner(zs(t, e)).assign_one((150, 120, 250, 280)) == ["T01", "E01"]


def test_center_and_hips_anchor():
    a = ZoneAssigner(zs(Zone(id="T01", polygon=SQ, anchor="center"), Zone(id="T02", polygon=SQ, anchor="hips")))
    assert a.assign_one((150, 150, 250, 250))[0] in ("T01", "T02")
    only_hips = ZoneAssigner(zs(Zone(id="T02", polygon=SQ, anchor="hips")))
    assert only_hips.assign_one((150, 0, 250, 200)) == ["T02"]  # 65% высоты = y 130 — внутри
    assert only_hips.assign_one((150, 0, 250, 100)) == []


def test_staff_area():
    z = Zone(id="S01", type="service_point", polygon=SQ, staff_polygon=[(300, 100), (400, 100), (400, 300), (300, 300)])
    a = ZoneAssigner(zs(z))
    assert a.in_staff_area((320, 100, 380, 250), "S01") and not a.in_staff_area((120, 100, 180, 250), "S01")


# ---------------------------------------------------------------- автопредложение
def _frames(tables, chairs=(), n=10, jitter=3.0, miss_every=0, noise=()):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        dets = []
        for b in tables:
            if miss_every and i % miss_every == 0:
                continue
            j = rng.normal(0, jitter, 4)
            dets.append(Detection(*(np.array(b) + j), 0.6, "dining table"))
        for b in chairs:
            dets.append(Detection(*b, 0.5, "chair"))
        if i == 0:
            dets += [Detection(*b, 0.4, "dining table") for b in noise]
        out.append(dets)
    return out


def test_cluster_keeps_stable_and_drops_one_off_detections():
    frames = _frames([(100, 100, 200, 160), (400, 300, 520, 380)], miss_every=4, noise=[(10, 400, 60, 440)])
    cl = cluster_boxes(frames, "dining table")
    assert len(cl) == 2
    assert cl[0].box[0] == pytest.approx(100, abs=6) or cl[1].box[0] == pytest.approx(100, abs=6)


def test_expand_table_includes_nearby_chairs_only():
    box = expand_table((100, 100, 200, 160), chairs=[(60, 110, 95, 170), (500, 500, 540, 560)])
    assert box[0] <= 60 and box[2] < 300 and box[3] < 300
    assert expand_table((100, 100, 200, 160), [])[0] == pytest.approx(80.0)  # ровно x1.4


def test_reading_order_rows_then_columns():
    boxes = [(400, 300, 480, 360), (100, 310, 180, 370), (300, 100, 380, 160), (50, 95, 130, 155)]
    assert reading_order(boxes) == [3, 2, 1, 0]


def test_suggest_numbers_tables_and_clips_to_frame():
    frames = _frames([(500, 300, 630, 400), (20, 30, 120, 90), (300, 40, 400, 100)],
                     chairs=[(125, 40, 160, 100)])
    z = suggest_from_detections(frames, (640, 480), "cam")
    assert [t.id for t in z.zones] == ["T01", "T02", "T03"]
    xs = [t.centroid()[0] for t in z.zones]
    assert xs[0] < xs[1] and z.zones[2].centroid()[1] > 250
    for t in z.zones:
        assert all(0 <= x <= 640 and 0 <= y <= 480 for x, y in t.polygon)
    assert z.zones[0].bbox()[2] >= 160  # стул рядом захвачен


def test_suggest_ignores_giant_table():
    assert suggest_from_detections(_frames([(0, 0, 630, 470)]), (640, 480)).zones == []


# ---------------------------------------------------------------- редактор без GUI
def test_editor_state_flow(tmp_path):
    st = EditorState(frame_size=(640, 480), save_path=tmp_path / "z.yaml", camera_id="cam")
    assert st.close_polygon() is None  # нет вершин
    for p in SQ:
        st.click(*p)
    st.handle_key(8)  # Backspace убирает вершину
    assert len(st.points) == 3
    st.handle_key(13)  # Enter закрывает полигон
    assert [z.id for z in st.zones.zones] == ["T01"] and st.points == []
    st.handle_key(ord("t"))
    assert st.zone_type == "entrance"
    for p in SQ:
        st.click(*p)
    st.handle_key(13)
    st.rename_last("Вход")
    assert st.zones.zones[-1].id == "E01" and st.zones.zones[-1].name == "Вход"
    st.handle_key(ord("s"))
    assert not st.dirty and len(load_zones(tmp_path / "z.yaml").zones) == 2
    st.handle_key(127)  # нет вершин -> удаляется последняя зона
    assert [z.id for z in st.zones.zones] == ["T01"]
    assert st.handle_key(ord("n")) == "rename" and st.handle_key(27) == "quit" and st.handle_key(-1) == ""


def test_editor_draw_does_not_need_window():
    from qonaq.zones.editor import draw_editor

    st = EditorState(frame_size=(320, 240))
    for p in [(10, 10), (100, 10), (100, 100)]:
        st.click(*p)
    img = draw_editor(np.zeros((240, 320, 3), np.uint8), st)
    assert img.shape == (240, 320, 3) and img.any()
