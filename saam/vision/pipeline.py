"""VisionPipeline: ONE camera loop shared by every vision mode.

    frame_source() -> BGR ndarray | None     (wrap saam/vision/camera.py)
    detector(frame) -> [{"label", "confidence", "bbox": [x1,y1,x2,y2]}, ...]
                       (wrap the existing SSD MobileNet V2 COCO inference)

Only this class touches the camera and the model. Find mode, counting,
scanning and annotation all read from the same processed frames, so nothing
else ever opens the webcam.

Controller-facing API (all return plain dicts/lists):
    get_detections()            one-shot detections (fresh or latest frame)
    count_objects(label)        count of a class, overlap-deduplicated
    start_find(target) / get_find_status() / stop_find()
    start_scan() / poll_scan_events() / stop_scan()
    stop()                      stop every mode and the camera loop NOW
    save_annotated_frame(path)  boxes + labels + confidence on latest frame
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

from .annotate import format_log_line, save_annotated_frame as _save_annotated
from .detection import dedupe, make_detection, matches_label, normalize_label
from .scanner import SceneTracker

log = logging.getLogger("saam.vision")

CAMERA_FAIL_LIMIT = 3


def enable_console_logging(level: int = logging.INFO) -> None:
    """Readable terminal logs for demos: labels, confidence, positions."""
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
    log.setLevel(level)


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    frame: object
    detections: List[dict]
    error: Optional[str] = None      # None | "camera_error" | "detector_error"


class VisionPipeline:
    def __init__(self, frame_source: Callable[[], object],
                 detector: Callable[[object], List[dict]], *,
                 fps: float = 3.0,
                 min_confidence: float = 0.5,
                 bbox_normalized: bool = False,
                 nms_iou: float = 0.5,
                 find_lost_after: int = 3,
                 scanner: Optional[SceneTracker] = None,
                 on_find_update: Optional[Callable[[dict], None]] = None,
                 on_scan_event: Optional[Callable[[dict], None]] = None,
                 log_frames: bool = True):
        self._frame_source = frame_source
        self._detector = detector
        self.fps = fps
        self.min_confidence = min_confidence
        self.bbox_normalized = bbox_normalized
        self.nms_iou = nms_iou
        self.find_lost_after = find_lost_after
        self.scanner = scanner or SceneTracker()
        self.on_find_update = on_find_update
        self.on_scan_event = on_scan_event
        self.log_frames = log_frames

        self._io_lock = threading.Lock()      # camera + model, one user at a time
        self._lock = threading.RLock()        # shared state
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0
        self._cam_failures = 0
        self._latest: Optional[FrameResult] = None
        self._find: Optional[dict] = None
        self._find_status: dict = self._idle_status()
        self._scan_active = False
        self._scan_events: Deque[dict] = deque(maxlen=50)

    # ================================================================
    # Core: one grab + detect + state update. Worker thread calls this;
    # tests can call it directly with no threads.
    # ================================================================
    def step(self) -> FrameResult:
        error = None
        dets: List[dict] = []
        with self._io_lock:
            frame = self._frame_source()
            if frame is not None:
                try:
                    dets = self._detect(frame)
                except Exception:                      # model failure must not kill the loop
                    log.exception("detector failed")
                    error = "detector_error"
        if frame is None:
            error = "camera_error"
        now = time.monotonic()

        find_snapshot = None
        scan_events: List[dict] = []
        with self._lock:
            self._frame_id += 1
            self._cam_failures = self._cam_failures + 1 if frame is None else 0
            result = FrameResult(self._frame_id, now, frame, dets, error)
            if frame is not None:
                self._latest = result
            if self._find is not None:
                find_snapshot = self._update_find(result)
            if self._scan_active and error is None:
                scan_events = self.scanner.update(dets, now)
                self._scan_events.extend(scan_events)

        if self.log_frames and error is None and (dets or self._find or self._scan_active):
            log.info(format_log_line(result.frame_id, dets))
        if error:
            log.warning("[frame %d] %s", result.frame_id, error)
        for ev in scan_events:
            log.info("SCAN %s: %s (%s)", ev["event"], ev["label"], ev["position"])
            self._safe_call(self.on_scan_event, ev)
        if find_snapshot is not None:
            self._safe_call(self.on_find_update, find_snapshot)
        return result

    def _detect(self, frame) -> List[dict]:
        h, w = frame.shape[:2]
        out = []
        for raw in self._detector(frame):
            if raw["confidence"] < self.min_confidence:
                continue
            x1, y1, x2, y2 = raw["bbox"]
            if not self.bbox_normalized:
                x1, x2, y1, y2 = x1 / w, x2 / w, y1 / h, y2 / h
            out.append(make_detection(normalize_label(raw["label"]),
                                      raw["confidence"], [x1, y1, x2, y2]))
        return dedupe(out, self.nms_iou)

    # ================================================================
    # One-shot queries
    # ================================================================
    def get_detections(self, max_age_s: float = 1.0) -> List[dict]:
        """Detections in the agreed format. Reuses the running loop's latest
        frame if it is fresh; otherwise processes one frame on demand."""
        with self._lock:
            latest = self._latest
            fresh = (latest is not None and self._worker_alive()
                     and time.monotonic() - latest.timestamp <= max_age_s)
        result = latest if fresh else self.step()
        return [dict(d) for d in result.detections]

    def count_objects(self, label: Optional[str] = None, max_age_s: float = 1.0) -> dict:
        """Count a class ('bottles', 'people' also accepted), or everything if
        label is None. Overlapping duplicates are already removed."""
        dets = self.get_detections(max_age_s)
        by_label: dict = {}
        for d in dets:
            by_label[d["label"]] = by_label.get(d["label"], 0) + 1
        if label is None:
            return {"label": None, "count": len(dets), "detections": dets, "by_label": by_label}
        matched = [d for d in dets if matches_label(d["label"], label)]
        return {"label": normalize_label(label), "count": len(matched),
                "detections": matched, "by_label": by_label}

    # ================================================================
    # Find mode
    # ================================================================
    def start_find(self, target: str) -> None:
        with self._lock:
            self._find = {"target": normalize_label(target), "misses": 0,
                          "ever_found": False, "was_centered": False, "event_latch": False}
            self._find_status = {**self._idle_status(), "active": True,
                                 "status": "searching", "target": self._find["target"]}
        self._ensure_worker()

    def get_find_status(self, consume_events: bool = True) -> dict:
        """Latest Find state. `centered_event` is True once per transition into
        the center (latched until read), so a slow poller cannot miss it."""
        with self._lock:
            status = dict(self._find_status)
            if consume_events and self._find is not None:
                self._find["event_latch"] = False
                self._find_status["centered_event"] = False
        return status

    def stop_find(self) -> None:
        with self._lock:
            self._find = None
            self._find_status = {**self._idle_status(), "status": "stopped"}
        self._halt_worker_if_idle()

    def _update_find(self, r: FrameResult) -> dict:
        f = self._find
        st = self._find_status
        if r.error == "camera_error" and self._cam_failures >= CAMERA_FAIL_LIMIT:
            st.update(status="camera_error", detected=False, centered=False)
        elif r.error == "detector_error":
            st.update(status="detector_error", detected=False, centered=False)
        elif r.error is None:
            hits = [d for d in r.detections if matches_label(d["label"], f["target"])]
            if hits:
                best = max(hits, key=lambda d: d["confidence"])
                f["misses"], f["ever_found"] = 0, True
                centered = best["position"] == "center"
                if centered and not f["was_centered"]:
                    f["event_latch"] = True
                f["was_centered"] = centered
                st.update(status="found", detected=True, position=best["position"],
                          confidence=best["confidence"], bbox=best["bbox"], centered=centered)
            else:
                f["misses"] += 1
                if f["misses"] >= self.find_lost_after:
                    f["was_centered"] = False
                    st.update(status="not_detected", detected=False, position=None,
                              confidence=None, bbox=None, centered=False)
        st["centered_event"] = f["event_latch"]
        st["frame_id"], st["timestamp"] = r.frame_id, r.timestamp
        return dict(st)

    @staticmethod
    def _idle_status() -> dict:
        return {"active": False, "status": "idle", "target": None, "detected": False,
                "position": None, "confidence": None, "bbox": None,
                "centered": False, "centered_event": False,
                "frame_id": None, "timestamp": None}

    # ================================================================
    # Scan mode
    # ================================================================
    def start_scan(self) -> None:
        with self._lock:
            self.scanner.reset()
            self._scan_events.clear()
            self._scan_active = True
        self._ensure_worker()

    def poll_scan_events(self) -> List[dict]:
        """Drain pending events: appeared / disappeared / moved."""
        with self._lock:
            events = list(self._scan_events)
            self._scan_events.clear()
        return events

    def stop_scan(self) -> None:
        with self._lock:
            self._scan_active = False
        self._halt_worker_if_idle()

    # ================================================================
    # Stop / visualization
    # ================================================================
    def stop(self) -> None:
        """Stop every mode and the camera loop immediately."""
        with self._lock:
            self._find = None
            self._find_status = {**self._idle_status(), "status": "stopped"}
            self._scan_active = False
        self._halt_worker()

    def save_annotated_frame(self, path: Optional[str] = None) -> Optional[str]:
        """Save the most recent frame with boxes/labels/confidence/position.
        Returns the path, or None if no frame has been processed yet."""
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        path = path or time.strftime("demo/annotated_%Y%m%d_%H%M%S.jpg")
        return _save_annotated(latest.frame, latest.detections, path)

    # ================================================================
    # Worker management
    # ================================================================
    def _worker_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker_alive():
                return
            self._stop_evt.clear()
            self._thread = threading.Thread(target=self._run, name="saam-vision", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        period = 1.0 / max(self.fps, 0.1)
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                self.step()
            except Exception:
                log.exception("vision loop error")
            self._stop_evt.wait(max(0.0, period - (time.monotonic() - t0)))

    def _halt_worker_if_idle(self) -> None:
        with self._lock:
            idle = self._find is None and not self._scan_active
        if idle:
            self._halt_worker()

    def _halt_worker(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None

    @staticmethod
    def _safe_call(fn, arg) -> None:
        if fn is None:
            return
        try:
            fn(arg)
        except Exception:
            log.exception("callback failed")