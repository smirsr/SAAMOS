"""Builds the spoken response string from an intent and a vision result.

Responses are heard, not read, so they are short and plain. All spoken wording
lives in this file so it can be changed in one place.

Inputs (see docs/integration-guide.md):
    intent        {"intent": "FIND_OBJECT", "target": "bottle"}
    vision_result FIND_OBJECT    -> detection dict (see detections.py for the format)
                  DESCRIBE_SCENE -> list of dicts from describe_scene(), or a
                                    dict with an "error" key if the camera failed
                  COUNT_OBJECTS  -> {"object": "bottle", "count": 2}
                  READ_TEXT      -> {"text": "..."} or {"error": "..."}
                  other intents  -> None

An empty list from describe_scene() means the camera worked and saw nothing.
A camera or model failure is reported as {"error": "..."} and is never
described as an empty scene.
"""

from typing import Dict, List, Optional

from .detections import label_of

FIND_OBJECT = "FIND_OBJECT"
DESCRIBE_SCENE = "DESCRIBE_SCENE"
COUNT_OBJECTS = "COUNT_OBJECTS"
READ_TEXT = "READ_TEXT"
STOP = "STOP"

CONFIDENCE_THRESHOLD = 0.5  # below this, a detection counts as "not found"
MAX_SCENE_OBJECTS = 3       # keep spoken scene descriptions short
MAX_READ_CHARS = 500        # longest text read aloud in one go

POSITION_PHRASES = {
    "left": "on the left",
    "right": "on the right",
    "center": "in front of you",
}

UNKNOWN_MESSAGE = "Sorry, I didn't catch that. Could you repeat it?"
STOP_MESSAGE = "Okay, stopping."
CAMERA_MESSAGE = "Sorry, I can't use the camera right now."
CAMERA_STOPPED_MESSAGE = "Sorry, I can't use the camera right now. I've stopped."
EMPTY_SCENE_MESSAGE = "I don't see any objects right now."

# --- Operating-mode messages ---
FIND_STOPPED_MESSAGE = "Okay, I've stopped searching."
NOT_SEARCHING_MESSAGE = "I'm not searching for anything right now."
SCAN_STARTED_MESSAGE = "Okay, I'll scan your surroundings and tell you what changes."
SCAN_STOPPED_MESSAGE = "Okay, I've stopped scanning."
NOT_SCANNING_MESSAGE = "I'm not scanning right now."
SCAN_TIMEOUT_MESSAGE = ("I've been scanning for a while, so I've stopped. "
                        "Say scan my surroundings to start again.")
LISTEN_TIMEOUT_MESSAGE = "I didn't hear a command."
LISTEN_FAILED_MESSAGE = "Sorry, I can't listen right now."
NOTHING_TO_REPEAT_MESSAGE = "I haven't said anything yet."
OCR_NOT_INSTALLED_MESSAGE = "Reading text isn't set up yet."
OCR_ERROR_MESSAGE = "Sorry, I can't read text right now."
NO_TEXT_MESSAGE = "I couldn't find any text."

_IRREGULAR_PLURALS = {"person": "people", "knife": "knives", "man": "men",
                      "woman": "women", "mouse": "mice"}


def _confidence(detection: Dict) -> float:
    value = detection.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def is_confident_detection(detection) -> bool:
    """True if the detection is a real find at or above the threshold."""
    return (
        isinstance(detection, dict)
        and detection.get("detected", True) is True        # new format has no flag
        and not detection.get("error")
        and _confidence(detection) >= CONFIDENCE_THRESHOLD
    )


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def plural(label: str, count: int) -> str:
    """"bottle" -> "bottles" (or the label itself when count is 1)."""
    if count == 1:
        return label
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    if label.endswith(("s", "x", "z", "ch", "sh")):
        return label + "es"
    if label.endswith("y") and len(label) > 1 and label[-2] not in "aeiou":
        return label[:-1] + "ies"
    return label + "s"


def unsupported_object_response(target: str) -> str:
    """Said when vision has no label for the requested object."""
    return "Sorry, I can't look for {} {} yet.".format(_article(target), target)


def _find_response(target: str, detection) -> str:
    # Anything that is not a proper detection dict, or that carries an error,
    # means vision failed. Do not say the object is missing in that case.
    if not isinstance(detection, dict) or detection.get("error"):
        return CAMERA_MESSAGE
    if not is_confident_detection(detection):
        return "I couldn't find your {}.".format(target)
    phrase = POSITION_PHRASES.get(detection.get("position"))
    if phrase is None:
        return "I can see your {}, but I'm not sure where it is.".format(target)
    return "Your {} is {}.".format(target, phrase)


def _scene_response(detections) -> str:
    # A list (even an empty one) is a successful scan. Anything else, such as
    # {"error": "CAMERA_UNAVAILABLE"}, None or garbage, means vision failed.
    if not isinstance(detections, list):
        return CAMERA_MESSAGE
    seen = [d for d in detections if is_confident_detection(d)]
    seen.sort(key=_confidence, reverse=True)
    parts = []
    for detection in seen[:MAX_SCENE_OBJECTS]:
        name = label_of(detection) or "object"
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


def count_response(target: str, result) -> str:
    """Answer "How many bottles do you see?" from {"object": ..., "count": n}."""
    if not isinstance(result, dict) or result.get("error"):
        return CAMERA_MESSAGE
    count = result.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return CAMERA_MESSAGE
    if count == 0:
        return "I don't see any {}.".format(plural(target, 0))
    if count == 1:
        return "I can see one {}.".format(target)
    return "I can see {} {}.".format(count, plural(target, count))


def read_text_response(result) -> str:
    """Speak the text found by OCR, or explain why there is none."""
    if not isinstance(result, dict):
        return OCR_ERROR_MESSAGE
    error = result.get("error")
    if error == "OCR_NOT_INSTALLED":
        return OCR_NOT_INSTALLED_MESSAGE
    if error:
        return OCR_ERROR_MESSAGE
    text = " ".join(str(result.get("text") or "").split())
    if not text:
        return NO_TEXT_MESSAGE
    return "It says: " + text[:MAX_READ_CHARS]


# --- Find Mode wording ---
def find_started_message(target: str) -> str:
    return "Okay, I'll look for your {}.".format(target)


def find_moved_message(target: str, position: str) -> str:
    phrase = POSITION_PHRASES.get(position, "in a different spot")
    return "Your {} is now {}.".format(target, phrase)


def find_lost_message(target: str) -> str:
    return "I lost sight of your {}.".format(target)


def find_timeout_message(target: str, seen_before: bool) -> str:
    if seen_before:
        return "I've stopped searching for your {}.".format(target)
    return "I couldn't find your {}. I've stopped searching.".format(target)


# --- Scan Mode wording ---
def scan_changes_message(previous: Dict[str, int], current: Dict[str, int],
                         positions: Dict[str, str]) -> str:
    """Describe what changed between two scene snapshots ({label: count})."""
    sentences = []
    for label in sorted(l for l in current if l not in previous):
        phrase = POSITION_PHRASES.get(positions.get(label))
        sentences.append("{} {} appeared{}.".format(
            _article(label).capitalize(), label, " " + phrase if phrase else ""))
    for label in sorted(l for l in previous if l not in current):
        sentences.append("The {} is out of sight.".format(label))
    for label in sorted(l for l in current if l in previous and current[l] != previous[l]):
        sentences.append("Now I see {} {}.".format(current[label], plural(label, current[label])))
    return " ".join(sentences[:MAX_SCENE_OBJECTS])


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
    if name == COUNT_OBJECTS:
        target = intent.get("target")
        if not target:
            return UNKNOWN_MESSAGE
        return count_response(str(target), vision_result)
    if name == READ_TEXT:
        return read_text_response(vision_result)
    return UNKNOWN_MESSAGE
