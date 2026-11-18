"""Unit tests for Find / count / scan / annotate using simulated detections.
No camera or model needed: run with `pytest tests/test_vision_modes.py`."""
import os
import numpy as np
import pytest
 
from saam.vision import VisionPipeline, SceneTracker
from saam.vision.detection import (dedupe, make_detection, position_from_bbox,
                                   matches_label)
 
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
 
 
def box(cx, w=0.1):
    return [cx - w / 2, 0.4, cx + w / 2, 0.6]
 
 
def raw(label, cx, conf=0.9):
    return {"label": label, "confidence": conf, "bbox": box(cx)}
 
 
class Scripted:
    """Detector that returns one scripted list per call (repeats the last)."""
    def __init__(self, frames):
        self.frames, self.i = frames, 0
 
    def __call__(self, frame):
        out = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return out
 
 
def make_pipe(frames, source=lambda: FRAME, **kw):
    return VisionPipeline(source, Scripted(frames), bbox_normalized=True,
                          log_frames=False, **kw)
 
 
# ---- detection format -------------------------------------------------
def test_position_left_center_right():
    assert position_from_bbox(box(0.15)) == "left"
    assert position_from_bbox(box(0.50)) == "center"
    assert position_from_bbox(box(0.85)) == "right"
 
 
def test_detection_format_and_supported_object():
    pipe = make_pipe([[raw("bottle", 0.2, 0.85)]])
    d = pipe.get_detections()[0]
    assert set(d) == {"label", "confidence", "position", "bbox"}
    assert d["label"] == "bottle" and d["position"] == "left" and d["confidence"] == 0.85
    assert all(0 <= v <= 1 for v in d["bbox"])
 
 
def test_pixel_bbox_is_normalized():
    pipe = VisionPipeline(lambda: FRAME,
                          Scripted([[{"label": "cup", "confidence": 0.8,
                                      "bbox": [320, 120, 384, 240]}]]),
                          log_frames=False)
    d = pipe.get_detections()[0]
    assert d["bbox"] == [0.5, 0.25, 0.6, 0.5] and d["position"] == "center"
 
 
def test_low_confidence_filtered():
    pipe = make_pipe([[raw("bottle", 0.5, 0.2)]], min_confidence=0.5)
    assert pipe.get_detections() == []
 
 
def test_no_detections():
    pipe = make_pipe([[]])
    assert pipe.get_detections() == []
    assert pipe.count_objects("bottle")["count"] == 0
 
 
# ---- counting ---------------------------------------------------------
def test_count_multiple_and_plural_and_people():
    pipe = make_pipe([[raw("bottle", 0.2), raw("bottle", 0.5), raw("bottle", 0.8),
                       raw("person", 0.4), raw("person", 0.7)]])
    assert pipe.count_objects("bottles")["count"] == 3
    assert pipe.count_objects("people")["count"] == 2
    assert pipe.count_objects(None)["by_label"] == {"bottle": 3, "person": 2}
 
 
def test_overlapping_duplicates_not_double_counted():
    a = raw("bottle", 0.50, 0.9)
    b = raw("bottle", 0.51, 0.7)      # near-identical box
    pipe = make_pipe([[a, b, raw("bottle", 0.85)]])
    assert pipe.count_objects("bottle")["count"] == 2
 
 
def test_label_matching_keeps_s_labels():
    assert matches_label("bus", "bus") and matches_label("scissors", "scissors")
    assert matches_label("bench", "benches")
 
 
# ---- find mode --------------------------------------------------------
def _find_pipe(frames, **kw):
    pipe = make_pipe(frames, **kw)
    with pipe._lock:                 # arm find state without the worker thread
        pipe._find = {"target": "bottle", "misses": 0, "ever_found": False,
                      "was_centered": False, "event_latch": False}
    return pipe
 
 
def test_find_tracks_position_and_centered_event_once():
    pipe = _find_pipe([[raw("bottle", 0.2)], [raw("bottle", 0.5)],
                       [raw("bottle", 0.5)], [raw("bottle", 0.8)]])
    pipe.step()
    s = pipe.get_find_status()
    assert s["status"] == "found" and s["position"] == "left" and not s["centered"]
    pipe.step()
    s = pipe.get_find_status()
    assert s["position"] == "center" and s["centered"] and s["centered_event"]
    pipe.step()                      # still centered: no second event
    s = pipe.get_find_status()
    assert s["centered"] and not s["centered_event"]
    pipe.step()
    assert pipe.get_find_status()["position"] == "right"
 
 
def test_find_not_detected_after_object_disappears():
    pipe = _find_pipe([[raw("bottle", 0.5)], [], [], []], find_lost_after=3)
    pipe.step()
    assert pipe.get_find_status()["status"] == "found"
    pipe.step(); pipe.step()
    assert pipe.get_find_status()["status"] == "found"      # brief flicker tolerated
    pipe.step()
    s = pipe.get_find_status()
    assert s["status"] == "not_detected" and s["detected"] is False and s["position"] is None
 
 
def test_find_camera_error():
    pipe = _find_pipe([[]], source=lambda: None)
    for _ in range(3):
        pipe.step()
    assert pipe.get_find_status()["status"] == "camera_error"
 
 
def test_stop_find_and_stop_all():
    pipe = make_pipe([[raw("bottle", 0.5)]], fps=50)
    pipe.start_find("bottle")
    assert pipe._worker_alive()
    pipe.stop_find()
    assert not pipe._worker_alive() and pipe.get_find_status()["status"] == "stopped"
    pipe.start_find("bottle"); pipe.start_scan()
    pipe.stop()
    assert not pipe._worker_alive()
    assert pipe.get_find_status()["status"] == "stopped" and not pipe._scan_active
 
 
def test_live_worker_updates_find():
    import time
    pipe = make_pipe([[raw("bottle", 0.2)]], fps=50)
    pipe.start_find("bottle")
    time.sleep(0.3)
    assert pipe.get_find_status()["position"] == "left"
    pipe.stop()
 
 
# ---- scanning ---------------------------------------------------------
def det(label, cx):
    return make_detection(label, 0.9, box(cx))
 
 
def run(tracker, seq, t0=0.0, dt=0.4):
    events = []
    for i, dets in enumerate(seq):
        events += tracker.update(dets, now=t0 + i * dt)
    return events
 
 
def test_appear_requires_persistence():
    tr = SceneTracker(persist_frames=3)
    assert run(tr, [[det("cup", 0.5)], [det("cup", 0.5)]]) == []       # only 2 frames
    ev = tr.update([det("cup", 0.5)], now=5)
    assert [e["event"] for e in ev] == ["appeared"]
 
 
def test_single_frame_flicker_ignored():
    tr = SceneTracker(persist_frames=3)
    assert run(tr, [[det("cup", 0.5)], [], [det("cup", 0.5)], [], []]) == []
 
 
def test_no_repeat_announcements_while_static():
    tr = SceneTracker(persist_frames=2)
    ev = run(tr, [[det("cup", 0.5)]] * 30)
    assert [e["event"] for e in ev] == ["appeared"]
 
 
def test_disappearance():
    tr = SceneTracker(persist_frames=2, absent_frames=3)
    ev = run(tr, [[det("cup", 0.5)]] * 3 + [[]] * 3)
    assert [e["event"] for e in ev] == ["appeared", "disappeared"]
 
 
def test_moved_between_positions():
    tr = SceneTracker(persist_frames=2, cooldown_s=0)
    seq = [[det("bottle", 0.15)]] * 3 + [[det("bottle", 0.30)]] + [[det("bottle", 0.5)]] * 4
    ev = run(tr, seq)
    kinds = [e["event"] for e in ev]
    assert kinds[0] == "appeared" and "moved" in kinds
    moved = [e for e in ev if e["event"] == "moved"][0]
    assert moved["from_position"] == "left" and moved["to_position"] == "center"
 
 
def test_cooldown_defers_moved_events():
    tr = SceneTracker(persist_frames=1, cooldown_s=10.0, move_threshold=0.2)
    ev = run(tr, [[det("bottle", 0.2)], [det("bottle", 0.5)], [det("bottle", 0.8)]], dt=1.0)
    assert [e["event"] for e in ev].count("moved") == 1       # second move held back
 
 
def test_pipeline_scan_start_poll_stop():
    pipe = make_pipe([[raw("cup", 0.5)]], scanner=SceneTracker(persist_frames=2))
    pipe.start_scan(); pipe._halt_worker()
    for _ in range(3):
        pipe.step()
    assert [e["event"] for e in pipe.poll_scan_events()] == ["appeared"]
    assert pipe.poll_scan_events() == []                    # drained
    pipe.stop_scan()
    pipe.step()
    assert pipe.poll_scan_events() == []
 
 
# ---- visualization ----------------------------------------------------
def test_annotated_frame_saved(tmp_path):
    pytest.importorskip("cv2")
    pipe = make_pipe([[raw("bottle", 0.2, 0.85)]])
    pipe.step()
    out = pipe.save_annotated_frame(str(tmp_path / "a.jpg"))
    assert os.path.getsize(out) > 0
 
 
def test_annotate_before_any_frame_returns_none():
    assert make_pipe([[]]).save_annotated_frame("x.jpg") is None
 
