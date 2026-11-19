"""Drawing and logging helpers for demo / report evidence.

Uses OpenCV only (already needed for camera capture); imported lazily so the
rest of saam.vision works without it.
"""
from __future__ import annotations

import os
from typing import List

from .detection import LEFT_EDGE, RIGHT_EDGE

_GREEN = (0, 255, 0)


def draw_detections(frame, detections: List[dict], show_zones: bool = True):
    """Return a copy of `frame` with boxes, labels, confidences, positions."""
    import cv2
    out = frame.copy()
    h, w = out.shape[:2]
    if show_zones:
        for edge in (LEFT_EDGE, RIGHT_EDGE):
            x = int(edge * w)
            cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(out, p1, p2, _GREEN, 2)
        text = f'{det["label"]} {det["confidence"]:.2f} {det["position"]}'
        (tw, th), base = cv2.getTextSize(text, font, 0.5, 1)
        ytxt = max(p1[1], th + base + 2)
        cv2.rectangle(out, (p1[0], ytxt - th - base - 2), (p1[0] + tw + 4, ytxt), _GREEN, -1)
        cv2.putText(out, text, (p1[0] + 2, ytxt - base - 1), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_annotated_frame(frame, detections: List[dict], path: str) -> str:
    import cv2
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if not cv2.imwrite(path, draw_detections(frame, detections)):
        raise IOError(f"could not write image to {path}")
    return path


def format_log_line(frame_id: int, detections: List[dict]) -> str:
    if not detections:
        return f"[frame {frame_id}] no detections"
    parts = [f'{d["label"]} {d["confidence"]:.2f} {d["position"]}' for d in detections]
    return f"[frame {frame_id}] " + " | ".join(parts)
