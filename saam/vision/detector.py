"""SSD MobileNet v2 (COCO) detector. The model loads on load()/first detect(), not on import."""
from pathlib import Path
 
import numpy as np
 
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "ssd_mobilenet_v2_coco_quant_postprocess.tflite"
DEFAULT_LABELS = ROOT / "models" / "coco_labels.txt"
 
 
class ModelLoadError(RuntimeError):
    """Missing model/labels/runtime, or a model with unexpected outputs."""
 
 
def horizontal_position(x_center):
    """x_center is normalized 0-1 across the image width."""
    if x_center < 1 / 3:
        return "left"
    if x_center > 2 / 3:
        return "right"
    return "center"
 
 
def load_labels(path):
    """Return {class_id: name}. Accepts '0  person' lines or one name per line."""
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    labels, plain = {}, []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1].strip()
        else:
            plain.append(line)
    if not labels:
        if plain and plain[0] == "???":   # TF labelmap placeholder; model ids start at person=0
            plain = plain[1:]
        labels = dict(enumerate(plain))
    return labels
 
 
def validate_labels(labels):
    if labels.get(0) != "person":
        raise ModelLoadError(
            f"Label file does not match the COCO model: id 0 is {labels.get(0)!r}, "
            "expected 'person'. Re-download coco_labels.txt from the same source as the model.")
    if "bottle" not in labels.values():
        raise ModelLoadError("Label file has no 'bottle' class.")
 
 
def _default_interpreter_factory(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as e:
            raise ModelLoadError(
                "No TFLite runtime found. Install ai-edge-litert (laptop) "
                "or tflite-runtime (board).") from e
    return Interpreter(model_path=str(model_path))
 
 
class ObjectDetector:
    def __init__(self, model_path=DEFAULT_MODEL, labels_path=DEFAULT_LABELS,
                 min_confidence=0.5, interpreter_factory=None):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.min_confidence = min_confidence
        self._factory = interpreter_factory or _default_interpreter_factory
        self._interp = None
 
    @property
    def is_loaded(self):
        return self._interp is not None
 
    def load(self):
        if self._interp is not None:
            return
        for path, what in ((self.model_path, "Model"), (self.labels_path, "Labels")):
            if not path.is_file():
                raise ModelLoadError(f"{what} file not found: {path}")
        try:
            import cv2
        except ImportError as e:
            raise ModelLoadError("OpenCV (cv2) is not installed.") from e
 
        labels = load_labels(self.labels_path)
        validate_labels(labels)
 
        interp = self._factory(self.model_path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        boxes_idx = count_idx = None
        rest = []
        for d in interp.get_output_details():
            shape = tuple(int(s) for s in d["shape"])
            if len(shape) == 3 and shape[-1] == 4:
                boxes_idx = d["index"]
            elif len(shape) == 1:
                count_idx = d["index"]
            else:
                rest.append(d["index"])
        if boxes_idx is None or len(rest) != 2:
            raise ModelLoadError("Unexpected model outputs; expected boxes, classes, scores, count.")
 
        self._cv = cv2
        self._labels = labels
        self._in_index = inp["index"]
        self._in_float = np.dtype(inp["dtype"]) != np.uint8
        _, self._h, self._w, _ = [int(s) for s in inp["shape"]]
        self._out = {"boxes": boxes_idx, "classes": rest[0], "scores": rest[1], "count": count_idx}
        self._interp = interp
 
    def detect(self, frame_bgr, min_confidence=None):
        """Return [{'object', 'confidence', 'box'(xmin,ymin,xmax,ymax 0-1), 'position'}]."""
        self.load()
        thr = self.min_confidence if min_confidence is None else min_confidence
        rgb = self._cv.cvtColor(frame_bgr, self._cv.COLOR_BGR2RGB)
        batch = np.expand_dims(self._cv.resize(rgb, (self._w, self._h)), 0)
        if self._in_float:
            batch = (batch.astype(np.float32) - 127.5) / 127.5
        else:
            batch = batch.astype(np.uint8)
        self._interp.set_tensor(self._in_index, batch)
        self._interp.invoke()
 
        boxes = self._interp.get_tensor(self._out["boxes"])[0]
        classes = self._interp.get_tensor(self._out["classes"])[0]
        scores = self._interp.get_tensor(self._out["scores"])[0]
        n = len(scores)
        if self._out["count"] is not None:
            n = min(n, int(self._interp.get_tensor(self._out["count"])[0]))
 
        results = []
        for i in range(n):
            if scores[i] < thr:
                continue
            ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
            results.append({
                "object": self._labels.get(int(classes[i]), "unknown"),
                "confidence": float(scores[i]),
                "box": (xmin, ymin, xmax, ymax),
                "position": horizontal_position((xmin + xmax) / 2),
            })
        return results
 
