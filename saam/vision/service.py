"""Controller-facing vision API: detect_object(target) and describe_scene()."""
from .camera import CameraError, CameraSystem
from .detector import ModelLoadError, ObjectDetector
 
ALIASES = {
    "water bottle": "bottle", "mug": "cup", "phone": "cell phone",
    "cellphone": "cell phone", "mobile phone": "cell phone", "computer": "laptop",
}
 
 
def normalize_target(target):
    name = (target or "").strip().lower()
    return ALIASES.get(name, name)
 
 
def _result(name, detected, position=None, confidence=0.0, error=None):
    r = {"object": name, "detected": detected, "position": position,
         "confidence": round(float(confidence), 3)}
    if error:
        r["error"] = error
    return r
 
 
class VisionService:
    def __init__(self, camera=None, detector=None, warmup_frames=3):
        self.camera = camera or CameraSystem()
        self.detector = detector or ObjectDetector()
        self.warmup_frames = warmup_frames
        self._ready = False
 
    def _grab(self):
        if not self._ready:
            self.camera.initialize_camera()
            for _ in range(self.warmup_frames):   # let auto-exposure settle
                self.camera.get_frame()
            self._ready = True
        ok, frame = self.camera.get_frame()
        if not ok:
            self.close()                          # re-open on the next request
            raise CameraError("Failed to read a frame from the camera.")
        return frame
 
    def detect_object(self, target):
        name = normalize_target(target)
        try:
            detections = self.detector.detect(self._grab())
        except CameraError:
            return _result(name, False, error="camera_error")
        except ModelLoadError:
            return _result(name, False, error="model_error")
        matches = [d for d in detections if d["object"] == name]
        if not matches:
            return _result(name, False)
        best = max(matches, key=lambda d: d["confidence"])
        return _result(name, True, best["position"], best["confidence"])
 
    def describe_scene(self):
        try:
            detections = self.detector.detect(self._grab())
        except CameraError:
            return {"objects": [], "count": 0, "error": "camera_error"}
        except ModelLoadError:
            return {"objects": [], "count": 0, "error": "model_error"}
        objs = [{"object": d["object"], "position": d["position"],
                 "confidence": round(d["confidence"], 3)}
                for d in sorted(detections, key=lambda d: -d["confidence"])]
        return {"objects": objs, "count": len(objs)}
 
    def close(self):
        self._ready = False
        self.camera.destroy_camera()
 
 
_default = None
 
 
def get_service():
    global _default
    if _default is None:
        _default = VisionService()
    return _default
 
 
def configure(service):
    """Swap the default service (tests, or a custom camera index on the board)."""
    global _default
    _default = service
 
 
def detect_object(target):
    return get_service().detect_object(target)
 
 
def describe_scene():
    return get_service().describe_scene()
 
