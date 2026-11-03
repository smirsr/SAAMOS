"""Manual demo (replaces detect_test.py).
   python -m saam.vision.demo [image.jpg]     # no argument = live camera, press q to quit
   SAAM_CAMERA=2 python -m saam.vision.demo    # choose a camera index or /dev/videoN
"""
import sys
import time
 
import cv2
 
from .camera import CameraError, CameraSystem
from .detector import ModelLoadError, ObjectDetector
 
 
def draw(frame, results):
    h, w = frame.shape[:2]
    for r in results:
        x1, y1, x2, y2 = r["box"]
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(frame, f'{r["object"]} {r["confidence"]:.2f} {r["position"]}',
                    (p1[0], max(p1[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame
 
 
def main():
    detector = ObjectDetector()
    try:
        detector.load()
        if len(sys.argv) > 1:
            frame = cv2.imread(sys.argv[1])
            if frame is None:
                sys.exit(f"Could not read image: {sys.argv[1]}")
            results = detector.detect(frame)
            for r in results:
                print(r)
            cv2.imshow("detections", draw(frame, results))
            cv2.waitKey(0)
            return
        cam = CameraSystem()
        cam.initialize_camera()
        try:
            while True:
                ok, frame = cam.get_frame()
                if not ok:
                    print("Frame capture failed")
                    break
                t = time.perf_counter()
                results = detector.detect(frame)
                print(f"{(time.perf_counter() - t) * 1000:.0f} ms, {len(results)} objects")
                if cam.display_frame(draw(frame, results), delay_ms=1):
                    break
        finally:
            cam.destroy_camera()
    except (ModelLoadError, CameraError) as e:
        sys.exit(str(e))
 
 
if __name__ == "__main__":
    main()
 
