"""Shared detection format for SAAMOS vision modes.

Every public vision function returns detections as plain dicts:

    {"label": "bottle", "confidence": 0.85, "position": "left",
     "bbox": [x1, y1, x2, y2]}          # normalized 0-1, origin top-left

No numpy types, no model-specific objects: safe to JSON-serialize and to pass
between the controller, audio and vision modules.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set

LEFT_EDGE = 1 / 3    # center x below this  -> "left"
RIGHT_EDGE = 2 / 3   # center x above this  -> "right"

_ALIASES = {
    "people": "person", "persons": "person",
    "phone": "cell phone", "cellphone": "cell phone", "mobile phone": "cell phone",
    "television": "tv", "plant": "potted plant",
}


def normalize_label(text: str) -> str:
    text = " ".join(text.lower().strip().split())
    return _ALIASES.get(text, text)


def label_candidates(query: str) -> Set[str]:
    """Labels a spoken query may refer to ('bottles' -> bottle).

    Plural stripping is done by candidate matching, not by rewriting, so
    COCO labels that end in 's' ('bus', 'scissors', 'skis') still match.
    """
    q = normalize_label(query)
    out = {q}
    if q.endswith("ies"):
        out.add(q[:-3] + "y")
    if q.endswith("es"):
        out.add(q[:-2])
    if q.endswith("s"):
        out.add(q[:-1])
    return out


def matches_label(det_label: str, query: str) -> bool:
    return det_label in label_candidates(query)


def position_from_bbox(bbox: Sequence[float],
                       left_edge: float = LEFT_EDGE,
                       right_edge: float = RIGHT_EDGE) -> str:
    cx = (bbox[0] + bbox[2]) / 2.0
    if cx < left_edge:
        return "left"
    if cx > right_edge:
        return "right"
    return "center"


def bbox_center(bbox: Sequence[float]) -> tuple:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def make_detection(label: str, confidence: float, bbox: Sequence[float]) -> dict:
    """Build one detection in the agreed format (bbox must be normalized)."""
    box = [round(min(1.0, max(0.0, float(v))), 4) for v in bbox]
    return {
        "label": label,
        "confidence": round(float(confidence), 3),
        "position": position_from_bbox(box),
        "bbox": box,
    }


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe(detections: Iterable[dict], iou_threshold: float = 0.5) -> List[dict]:
    """Per-label greedy NMS: keep the most confident of overlapping boxes."""
    kept: List[dict] = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        if any(k["label"] == det["label"] and iou(k["bbox"], det["bbox"]) >= iou_threshold
               for k in kept):
            continue
        kept.append(det)
    return kept