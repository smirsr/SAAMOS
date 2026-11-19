"""Normalizes what the vision module returns, so the rest of the code sees one shape.

Agreed detection format (from the vision owner):

    {"label": "bottle", "confidence": 0.85, "position": "left", "bbox": [x1, y1, x2, y2]}

    label       the object name (same vocabulary as object_labels.txt)
    confidence  0.0 to 1.0
    position    "left", "center" or "right"
    bbox        pixel box [x1, y1, x2, y2]; kept as is, the controller does not need it

The earlier format {"object": "bottle", "detected": True, ...} is still accepted.

What each vision function may return
------------------------------------
    detect_object(target)   one detection dict (or a list of them, best match is used)
                            None, {} or [] when the object is simply not there
                            {"error": "..."} if the camera or model failed
    describe_scene()        a list of detection dicts ([] = camera worked, saw nothing)
                            {"error": "..."} if the camera or model failed
    count_objects(target)   {"label": "bottle", "count": 2} or {"error": "..."}

"Not found" and "failed" are different on purpose: a failure must never be
reported to the user as "your bottle is not here".

After normalizing, every detection has BOTH "label" and "object" (the same
value) and a "detected" flag, so older code keeps working.
"""

import logging
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def label_of(detection) -> Optional[str]:
    """The object name of a detection, whichever key it uses."""
    if not isinstance(detection, dict):
        return None
    label = detection.get("label")
    if label is None:
        label = detection.get("object")
    return None if label is None else str(label)


def canonical(detection: Dict) -> Dict:
    """Copy a detection and add the fields the rest of the code relies on."""
    result = dict(detection)
    label = label_of(detection)
    if label is not None:
        result["label"] = label
        result["object"] = label
    result.setdefault("detected", True)       # new format has no flag: present means detected
    return result


def not_found(target: str) -> Dict:
    return {"label": target, "object": target, "detected": False,
            "position": None, "confidence": 0.0}


def _confidence(detection: Dict) -> float:
    value = detection.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def normalize_find_result(raw, target: str) -> Dict:
    """Turn whatever detect_object(target) returned into one detection dict."""
    if raw is None or raw == {} or raw == []:
        return not_found(target)
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("error"):
                return item
        candidates = [canonical(item) for item in raw
                      if isinstance(item, dict) and label_of(item) in (None, target)]
        if not candidates:
            return not_found(target)
        return max(candidates, key=_confidence)
    if isinstance(raw, dict):
        if raw.get("error"):
            return raw
        label = label_of(raw)
        # A detection of a different object is not an answer to this question.
        if label is not None and label.strip().lower() != str(target).strip().lower():
            LOGGER.warning("vision answered for %r when asked about %r; treating as not found",
                           label, target)
            return not_found(target)
        return canonical(raw)
    return {"label": target, "object": target, "detected": False, "error": "VISION_ERROR"}


def normalize_scene_result(raw):
    """Turn whatever describe_scene() returned into a list of detections, or an error dict."""
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("error"):
                return item
        return [canonical(item) for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict) and raw.get("error"):
        return raw
    return {"error": "VISION_ERROR"}          # None or garbage: vision did not really answer
