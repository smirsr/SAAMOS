"""Builds the spoken response string from an intent and a vision result.

Responses are heard, not read, so they are short and plain.

Inputs (see docs/integration-guide.md):
    intent        {"intent": "FIND_OBJECT", "target": "bottle"}
    vision_result FIND_OBJECT    -> dict from detect_object()
                  DESCRIBE_SCENE -> list of dicts from describe_scene()
                  STOP/UNKNOWN   -> None
"""

from typing import Dict, List, Optional

FIND_OBJECT = "FIND_OBJECT"
DESCRIBE_SCENE = "DESCRIBE_SCENE"
STOP = "STOP"

CONFIDENCE_THRESHOLD = 0.5  # below this, a detection counts as "not found"
MAX_SCENE_OBJECTS = 3       # keep spoken scene descriptions short

POSITION_PHRASES = {
    "left": "on the left",
    "right": "on the right",
    "center": "in front of you",
}

UNKNOWN_MESSAGE = "Sorry, I didn't catch that. Could you repeat it?"
STOP_MESSAGE = "Okay, stopping."
CAMERA_MESSAGE = "Sorry, I can't use the camera right now."
EMPTY_SCENE_MESSAGE = "I don't see any objects right now."


def _confidence(detection: Dict) -> float:
    value = detection.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def is_confident_detection(detection) -> bool:
    """True if the detection is a real find at or above the threshold."""
    return (
        isinstance(detection, dict)
        and detection.get("detected") is True
        and not detection.get("error")
        and _confidence(detection) >= CONFIDENCE_THRESHOLD
    )


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _find_response(target: str, detection) -> str:
    if isinstance(detection, dict) and detection.get("error"):
        return CAMERA_MESSAGE
    if not is_confident_detection(detection):
        return "I couldn't find your {}.".format(target)
    phrase = POSITION_PHRASES.get(detection.get("position"))
    if phrase is None:
        return "I can see your {}, but I'm not sure where it is.".format(target)
    return "Your {} is {}.".format(target, phrase)


def _scene_response(detections) -> str:
    if not isinstance(detections, list):
        detections = []
    seen = [d for d in detections if is_confident_detection(d)]
    seen.sort(key=_confidence, reverse=True)
    parts = []
    for detection in seen[:MAX_SCENE_OBJECTS]:
        name = str(detection.get("object", "object"))
        part = "{} {}".format(_article(name), name)
        phrase = POSITION_PHRASES.get(detection.get("position"))
        if phrase:
            part += " " + phrase
        parts.append(part)
    if not parts:
        return EMPTY_SCENE_MESSAGE
    if len(parts) == 1:
        listed = parts[0]
    elif len(parts) == 2:
        listed = parts[0] + " and " + parts[1]
    else:
        listed = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return "I can see {}.".format(listed)


def build_response(intent: Dict, vision_result=None) -> str:
    """Return the sentence to speak. Never raises on odd input."""
    name = intent.get("intent") if isinstance(intent, dict) else None
    if name == STOP:
        return STOP_MESSAGE
    if name == FIND_OBJECT:
        target = intent.get("target")
        if not target:
            return UNKNOWN_MESSAGE
        return _find_response(str(target), vision_result)
    if name == DESCRIBE_SCENE:
        return _scene_response(vision_result)
    return UNKNOWN_MESSAGE
