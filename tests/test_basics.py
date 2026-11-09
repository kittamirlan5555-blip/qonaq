"""M0/M1: конфиг, источник видео, детектор-заглушка, блюр, CLI."""
import numpy as np
import pytest
import yaml

from qonaq.config import SiteConfig, expand_env, load_site, mask_secrets, site_for_file
from qonaq.detect.base import Detection, Detector, FakeDetector
from qonaq.detect.onnx_yolox import nms
from qonaq.render.blur import blur_people, head_region
from qonaq.sources.video import VideoSource, backoff_delays, parse_source


def test_site_config_defaults_match_spec():
    s = SiteConfig()
    th = s.thresholds
    assert (th.seat_min_s, th.visit_min_s, th.leave_min_s, th.clean_min_s, th.dropout_tolerance_s) == (15, 4, 90, 10, 5)
    assert (s.alerts.no_waiter_s, s.alerts.dirty_table_s, s.alerts.long_service_without_visit_s) == (180, 300, 900)
    assert s.privacy.blur_faces is True


def test_site_config_rejects_unknown_keys_and_raw_video(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump({"thresholds": {"seat_min": 3}}))
    with pytest.raises(ValueError):
        load_site(p)
    p.write_text(yaml.safe_dump({"privacy": {"save_raw_video": True}}))
    with pytest.raises(ValueError):
        load_site(p)


def test_site_yaml_roundtrip(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text(yaml.safe_dump({
        "site_id": "x", "profile": "clinic", "process_fps": 2,
        "cameras": [{"id": "c1", "source": "webcam:0"}, {"id": "c2", "source": "rtsp://u:${PW}@1.2.3.4/s"}],
    }))
    s = load_site(p)
    assert s.profile == "clinic" and [c.id for c in s.cameras] == ["c1", "c2"]


def test_duplicate_camera_ids_rejected():
    with pytest.raises(ValueError):
        SiteConfig(cameras=[{"id": "a", "source": "webcam:0"}, {"id": "a", "source": "webcam:1"}])


def test_env_expansion_and_secret_masking(monkeypatch):
    monkeypatch.setenv("CAM_PASS", "s3cret")
    url = expand_env("rtsp://admin:${CAM_PASS}@10.0.0.2:554/Streaming/Channels/102")
    assert "s3cret" in url
    assert "s3cret" not in mask_secrets(url) and "admin:***@" in mask_secrets(url)


def test_parse_source():
    assert parse_source("webcam:1") == ("webcam", 1)
    assert parse_source("0") == ("webcam", 0)
    assert parse_source("rtsp://a/b")[0] == "stream"
    assert parse_source("data/videos/x.mp4") == ("file", "data/videos/x.mp4")


def test_backoff_is_exponential_and_capped():
    g = backoff_delays(1, 2, 10)
    assert [next(g) for _ in range(6)] == [1, 2, 4, 8, 10, 10]


def test_site_for_file():
    s = site_for_file("data/videos/abc.mp4")
    assert s.cameras[0].id == "abc" and s.cameras[0].source.endswith("abc.mp4")


@pytest.fixture()
def tiny_video(tmp_path):
    """Синтетический ролик 4 с, 20 fps: номер кадра закодирован яркостью."""
    import cv2

    path = tmp_path / "tiny.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    for i in range(80):
        w.write(np.full((48, 64, 3), i * 3, np.uint8))
    w.release()
    return path


def test_file_source_samples_to_process_fps(tiny_video):
    with VideoSource(str(tiny_video), process_fps=5) as src:
        items = list(src)
    ts = [t for _, t in items]
    assert 18 <= len(items) <= 21
    assert ts[0] == pytest.approx(0.0, abs=0.06)
    assert np.allclose(np.diff(ts), 0.2, atol=0.06)  # время сцены, а не номера кадров


def test_file_source_max_seconds_and_loop(tiny_video):
    with VideoSource(str(tiny_video), process_fps=5, max_seconds=2.0) as src:
        assert max(t for _, t in src) <= 2.0
    with VideoSource(str(tiny_video), process_fps=5, loop=True, max_seconds=9.0) as src:
        ts = [t for _, t in src]
    assert ts[-1] > 8.0 and all(b > a for a, b in zip(ts, ts[1:], strict=False))  # время монотонно через «склейку»


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoSource("no/such/file.mp4").open()


def test_stream_source_reconnects_with_backoff():
    """Обрыв RTSP: источник переподключается, задержки растут экспоненциально."""

    class FakeCap:
        def __init__(self, ok, frames=5):
            self.ok, self.frames = ok, frames

        def isOpened(self):
            return self.ok

        def read(self):
            if self.frames > 0:
                self.frames -= 1
                return True, np.zeros((4, 4, 3), np.uint8)
            return False, None

        def get(self, _):
            return 0

        def set(self, *_):
            return True

        def release(self):
            pass

    attempts, sleeps = [], []
    plan = [True, False, False, True]  # подключились, после обрыва — две неудачи, затем успех

    def opener():
        ok = plan[len(attempts)] if len(attempts) < len(plan) else True
        attempts.append(ok)
        return FakeCap(ok, frames=5 if len(attempts) == 1 else 10_000)

    def fake_sleep(d):
        sleeps.append(d)
        return False

    src = VideoSource("rtsp://user:pw@cam/stream", process_fps=50, opener=opener, sleep=fake_sleep)
    src.open()
    try:
        import time

        deadline = time.time() + 5
        while src.reconnects < 1 and time.time() < deadline:
            src.read()
        assert src.reconnects == 1
        assert sleeps[:2] == [1.0, 2.0]
        assert src.read() is not None
    finally:
        src.close()


def test_fake_detector_satisfies_interface():
    d = FakeDetector([[Detection(0, 0, 10, 20, 0.9, "person")]])
    assert isinstance(d, Detector)
    out = d.detect([np.zeros((10, 10, 3), np.uint8)] * 2)
    assert len(out) == 2 and out[0][0].label == "person"


def test_nms_drops_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), 0.5)
    assert sorted(keep) == [0, 2]


def test_blur_changes_head_only():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    orig = img.copy()
    box = (50.0, 20.0, 110.0, 190.0)
    blur_people(img, [box])
    x1, y1, x2, y2 = head_region(box)
    assert not np.array_equal(img[y1:y2, x1:x2], orig[y1:y2, x1:x2])
    assert np.array_equal(img[y2 + 5:, :], orig[y2 + 5:, :])  # ниже головы кадр не тронут
    # в размытой области не осталось мелких деталей
    assert img[y1:y2, x1:x2].astype(float).std() < orig[y1:y2, x1:x2].astype(float).std() * 0.6


def test_cli_parser_has_commands():
    from qonaq.cli import HANDLERS, build_parser

    ap = build_parser()
    assert ap.parse_args(["doctor", "--no-model"]).cmd == "doctor"
    assert {"doctor", "run-file"} <= set(HANDLERS)
