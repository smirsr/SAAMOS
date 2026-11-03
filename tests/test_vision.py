"""Vision tests. No webcam, no model file, no TFLite runtime needed.
Run from the repo root:  python -m pytest tests/test_vision.py
"""
import numpy as np
import pytest
 
from saam.vision import (CameraError, CameraSystem, ModelLoadError, ObjectDetector,
                         VisionService, configure, describe_scene, detect_object)
from saam.vision.detector import horizontal_position, load_labels
from saam.vision.camera import resolve_camera_source
 
 
# ---------- fakes ----------
class FakeCapture:
    def __init__(self, frames=None, opened=True):
        self.frames = list(frames if frames is not None else [])
        self.opened = opened
        self.released = False
 
    def isOpened(self):
        return self.opened
 
    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)
 
    def release(self):
        self.released = True
 
 
def blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)
 
 
class FakeInterpreter:
    """Mimics the TFLite interpreter for SSD MobileNet postprocess models."""
    def __init__(self, detections):  # [(class_id, score, (ymin, xmin, ymax, xmax)), ...]
        n = 10
        boxes, classes, scores = np.zeros((1, n, 4)), np.zeros((1, n)), np.zeros((1, n))
        for i, (c, s, b) in enumerate(detections):
            boxes[0, i], classes[0, i], scores[0, i] = b, c, s
        self.out = {0: boxes, 1: classes, 2: scores, 3: np.array([len(detections)], dtype=float)}
 
    def allocate_tensors(self): pass
    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, 300, 300, 3]), "dtype": np.uint8}]
    def get_output_details(self):
        shapes = [(1, 10, 4), (1, 10), (1, 10), (1,)]
        return [{"index": i, "shape": np.array(s)} for i, s in enumerate(shapes)]
    def set_tensor(self, index, value): self.last_input = value
    def invoke(self): pass
    def get_tensor(self, index): return self.out[index]
 
 
@pytest.fixture
def files(tmp_path):
    model = tmp_path / "m.tflite"
    model.write_bytes(b"x")
    labels = tmp_path / "labels.txt"
    labels.write_text("person\nbicycle\ncar\nbottle\n")
    return model, labels
 
 
def make_detector(files, detections):
    model, labels = files
    return ObjectDetector(model, labels, interpreter_factory=lambda p: FakeInterpreter(detections))
 
 
class StubDetector:
    def __init__(self, result=None, error=None):
        self.result, self.error = result or [], error
 
    def detect(self, frame):
        if self.error:
            raise self.error
        return self.result
 
 
def make_service(frames, detector, opened=True):
    cam = CameraSystem(camera=0, capture_factory=lambda *a: FakeCapture(frames, opened))
    return VisionService(camera=cam, detector=detector, warmup_frames=0)
 
 
# ---------- camera ----------
def test_camera_accepts_one_argument():
    assert CameraSystem(0).Key == "q"
 
 
def test_failed_capture_returns_false_none_without_flipping():
    cam = CameraSystem(0, mirror=True, capture_factory=lambda *a: FakeCapture([]))
    cam.initialize_camera()
    assert cam.get_frame() == (False, None)
 
 
def test_get_frame_before_initialize_is_safe():
    assert CameraSystem(0).get_frame() == (False, None)
 
 
def test_unopened_camera_raises_camera_error():
    cam = CameraSystem(0, capture_factory=lambda *a: FakeCapture(opened=False))
    with pytest.raises(CameraError):
        cam.initialize_camera()
 
 
def test_mirror_flag_controls_flip():
    f = blank(); f[:, 0] = 255
    plain = CameraSystem(0, capture_factory=lambda *a: FakeCapture([f.copy()]))
    plain.initialize_camera()
    assert plain.get_frame()[1][0, 0, 0] == 255
    mirrored = CameraSystem(0, mirror=True, capture_factory=lambda *a: FakeCapture([f.copy()]))
    mirrored.initialize_camera()
    assert mirrored.get_frame()[1][0, -1, 0] == 255
 
 
def test_camera_source_resolution(monkeypatch):
    monkeypatch.delenv("SAAM_CAMERA", raising=False)
    assert resolve_camera_source() == 0
    assert resolve_camera_source("2") == 2
    assert resolve_camera_source("/dev/video3") == "/dev/video3"
    monkeypatch.setenv("SAAM_CAMERA", "1")
    assert resolve_camera_source() == 1
 
 
def test_destroy_without_initialize_is_safe():
    CameraSystem(0).destroy_camera()
 
 
def test_destroy_releases_camera():
    cap = FakeCapture([blank()])
    cam = CameraSystem(0, capture_factory=lambda *a: cap)
    cam.initialize_camera()
    cam.destroy_camera()
    assert cap.released
 
 
# ---------- detector ----------
def test_detector_does_not_load_on_construction(files):
    assert not make_detector(files, []).is_loaded
 
 
def test_missing_model_raises_model_load_error(tmp_path):
    with pytest.raises(ModelLoadError):
        ObjectDetector(tmp_path / "nope.tflite", tmp_path / "nope.txt").load()
 
 
def test_label_file_with_wrong_order_is_rejected(files):
    model, labels = files
    labels.write_text("bicycle\nperson\ncar\nbottle\n")
    with pytest.raises(ModelLoadError):
        ObjectDetector(model, labels, interpreter_factory=lambda p: FakeInterpreter([])).load()
 
 
def test_load_labels_formats(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0  person\n1  bicycle\n")
    assert load_labels(p) == {0: "person", 1: "bicycle"}
    p.write_text("???\nperson\nbicycle\n")
    assert load_labels(p) == {0: "person", 1: "bicycle"}
    p.write_text("person\nbicycle\n")
    assert load_labels(p) == {0: "person", 1: "bicycle"}
 
 
def test_horizontal_position():
    assert horizontal_position(0.1) == "left"
    assert horizontal_position(0.5) == "center"
    assert horizontal_position(0.9) == "right"
 
 
def test_detect_maps_labels_positions_and_threshold(files):
    det = make_detector(files, [
        (3, 0.90, (0.2, 0.1, 0.5, 0.3)),    # bottle, left
        (2, 0.30, (0.2, 0.4, 0.5, 0.6)),    # car below threshold
    ])
    out = det.detect(blank())
    assert len(out) == 1
    assert out[0]["object"] == "bottle" and out[0]["position"] == "left"
 
 
# ---------- service / controller interface ----------
BOTTLE_RIGHT = [{"object": "bottle", "confidence": 0.8, "position": "right", "box": (0, 0, 1, 1)}]
 
 
def test_detect_object_found():
    svc = make_service([blank()], StubDetector(BOTTLE_RIGHT))
    assert svc.detect_object("bottle") == {
        "object": "bottle", "detected": True, "position": "right", "confidence": 0.8}
 
 
def test_detect_object_alias():
    svc = make_service([blank()], StubDetector(BOTTLE_RIGHT))
    assert svc.detect_object("Water Bottle")["detected"] is True
 
 
def test_detect_object_not_found():
    r = make_service([blank()], StubDetector([])).detect_object("bottle")
    assert r["detected"] is False and r["position"] is None and "error" not in r
 
 
def test_detect_object_camera_error_when_not_opened():
    r = make_service([], StubDetector(), opened=False).detect_object("bottle")
    assert r["detected"] is False and r["error"] == "camera_error"
 
 
def test_detect_object_camera_error_on_failed_read():
    r = make_service([], StubDetector()).detect_object("bottle")
    assert r["error"] == "camera_error"
 
 
def test_detect_object_model_error():
    r = make_service([blank()], StubDetector(error=ModelLoadError("x"))).detect_object("bottle")
    assert r["error"] == "model_error"
 
 
def test_describe_scene_returns_list_sorted_by_confidence():
    dets = [{"object": "cup", "confidence": 0.6, "position": "left", "box": (0, 0, 1, 1)}] + BOTTLE_RIGHT
    r = make_service([blank()], StubDetector(dets)).describe_scene()
    assert isinstance(r, list)
    assert r == [{"object": "bottle","detected": True, "position": "right","confidence": 0.8},
                 {"object": "cup","detected": True, "position": "left", "confidence": 0.6}]
 
 
def test_describe_scene_empty_scan_returns_empty_list():
    r = make_service([blank()], StubDetector([])).describe_scene()
    assert r == [] and isinstance(r, list)
 
 
def test_describe_scene_camera_error():
    r = make_service([], StubDetector(), opened=False).describe_scene()
    assert r == {"error": "camera_error"}
 
 
def test_describe_scene_model_error():
    r = make_service([blank()], StubDetector(error=ModelLoadError("x"))).describe_scene()
    assert r == {"error": "model_error"}
 
 
def test_module_level_functions_use_configured_service():
    configure(make_service([blank(), blank()], StubDetector(BOTTLE_RIGHT)))
    assert detect_object("bottle")["detected"] is True
    assert len(describe_scene()) == 1
    configure(None)
 
