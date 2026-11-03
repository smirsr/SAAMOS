import sys
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

MODEL = "models/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
LABELS = "models/coco_labels.txt"

def load_labels(path):
    labels = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():   # "9  traffic light"
                labels[int(parts[0])] = parts[1]
            else:                                        # "traffic light"
                labels[i] = line
    return labels

labels = load_labels(LABELS)
interp = Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
outs = interp.get_output_details()
_, H, W, _ = inp["shape"]

def position(x_center_norm):
    if x_center_norm < 0.33: return "left"
    if x_center_norm > 0.66: return "right"
    return "center"

def detect(frame_bgr, min_conf=0.5):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (W, H))
    interp.set_tensor(inp["index"], np.expand_dims(resized, 0).astype(np.uint8))
    interp.invoke()
    boxes   = interp.get_tensor(outs[0]["index"])[0]   # ymin, xmin, ymax, xmax (0-1)
    classes = interp.get_tensor(outs[1]["index"])[0]
    scores  = interp.get_tensor(outs[2]["index"])[0]
    results = []
    for box, cls, score in zip(boxes, classes, scores):
        if score < min_conf:
            continue
        ymin, xmin, ymax, xmax = box
        results.append({
            "label": labels.get(int(cls), "unknown"),
            "confidence": float(score),
            "box": (xmin, ymin, xmax, ymax),
            "position": position((xmin + xmax) / 2),
        })
    return results

def draw(frame, results):
    h, w = frame.shape[:2]
    for r in results:
        x1, y1, x2, y2 = r["box"]
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(frame, f'{r["label"]} {r["confidence"]:.2f} {r["position"]}',
                    (p1[0], max(p1[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

if __name__ == "__main__":
    if len(sys.argv) > 1:                      # python detect_test.py photo.jpg
        frame = cv2.imread(sys.argv[1])
        res = detect(frame)
        for r in res:
            print(r)
        cv2.imshow("detections", draw(frame, res))
        cv2.waitKey(0)
    else:                                      # live camera, press q to quit
        cap = cv2.VideoCapture(1)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("detections", draw(frame, detect(frame)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
    cv2.destroyAllWindows()