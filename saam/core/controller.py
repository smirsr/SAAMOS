"""Main controller: recognized text -> intent -> vision (if needed) -> response.

handle_command(text) is the original one-shot entry point described in
docs/integration-guide.md. It is unchanged in behavior and still works on its
own. Until the real vision module is ready, it uses MockVision so the
controller can be run and tested without hardware.

The new operating modes (Find Mode, Scan Mode, push-to-talk, repeat, ...) live
in modes.py and are built on top of respond_to_intent() below, so both paths
share the same wording and the same vision-failure handling.

Run the demo from the repository root:

    python -m saam.core.controller
    python -m saam.core.controller --interactive

How STOP reaches the audio side
-------------------------------
The controller does not own the speaker or the microphone. When the user says
stop, the on_stop callback it was given is called. The audio owner passes in a
function (proposed name: stop_speaking) that cuts off any speech that is
playing. See "STOP and the audio loop" in the integration guide.
"""

import argparse
import logging
from typing import Callable, Dict, List, Optional

from .detections import label_of, normalize_find_result, normalize_scene_result
from .intent_parser import (DESCRIBE_SCENE, FIND_OBJECT, START_FIND, STOP,
                            STOP_FIND, STOP_SCAN, parse_intent)
from .object_labels import is_supported_label
from .response_builder import (UNKNOWN_MESSAGE, build_response,
                               unsupported_object_response)

LOGGER = logging.getLogger(__name__)

# Simulated vision results, in the agreed detection format (see detections.py).
MOCK_SCENE = [
    {"label": "bottle", "confidence": 0.91, "position": "left", "bbox": [40, 120, 160, 400]},
    {"label": "cup", "confidence": 0.84, "position": "right", "bbox": [480, 200, 600, 360]},
    {"label": "laptop", "confidence": 0.77, "position": "center", "bbox": [220, 160, 420, 330]},
]

# Where a mock object sits in a 640 x 480 frame, for add().
_MOCK_BOXES = {"left": [40, 150, 160, 350], "center": [260, 150, 380, 350],
               "right": [480, 150, 600, 350]}


class MockVision:
    """Stand-in for the vision module, with the same two functions.

    It answers in the agreed detection format, and returns None from
    detect_object() when the object is not there, like the real module.
    add / remove / set_position let a test or a demo change the scene while the
    controller is running, to simulate objects appearing, leaving or moving.
    """

    def __init__(self, scene: Optional[List[Dict]] = None):
        self.scene = [dict(item) for item in (MOCK_SCENE if scene is None else scene)]

    def detect_object(self, target: str) -> Optional[Dict]:
        for item in self.scene:
            if label_of(item) == target:
                return dict(item)
        return None                      # not in view (this is not an error)

    def describe_scene(self) -> List[Dict]:
        return [dict(item) for item in self.scene]

    def add(self, label: str, position: str = "center", confidence: float = 0.9) -> None:
        self.scene.append({"label": label, "confidence": confidence, "position": position,
                           "bbox": list(_MOCK_BOXES.get(position, _MOCK_BOXES["center"]))})

    def remove(self, label: str) -> None:
        self.scene = [item for item in self.scene if label_of(item) != label]

    def set_position(self, label: str, position: str) -> None:
        for item in self.scene:
            if label_of(item) == label:
                item["position"] = position
                item["bbox"] = list(_MOCK_BOXES.get(position, _MOCK_BOXES["center"]))


def respond_to_intent(
    intent: Dict,
    detect_object: Callable[[str], Dict],
    describe_scene: Callable[[], List[Dict]],
    on_stop: Optional[Callable[[], None]] = None,
) -> str:
    """Answer one already-parsed intent once, and return the response string.

    Handles the original intents (FIND_OBJECT, DESCRIBE_SCENE, STOP). Without
    the mode controller there is nothing to keep running, so:
      START_FIND           is answered once, like FIND_OBJECT
      STOP_FIND / STOP_SCAN behave like STOP
      anything else new    is "not understood" here (use ModeController)
    Never raises: a failing module produces a spoken error message instead.
    A vision failure (exception, error dict, or bad return value) is reported
    as a camera problem, never as "nothing found" or an empty scene.
    """
    name = intent["intent"]
    if name == START_FIND:
        intent = {"intent": FIND_OBJECT, "target": intent["target"]}
        name = FIND_OBJECT
    elif name in (STOP_FIND, STOP_SCAN):
        intent = {"intent": STOP, "target": None}
        name = STOP
    elif name not in (FIND_OBJECT, DESCRIBE_SCENE, STOP):
        return UNKNOWN_MESSAGE

    vision_result = None
    if name == STOP:
        if on_stop is not None:
            try:
                on_stop()
            except Exception:
                LOGGER.exception("on_stop callback failed")
    elif name == FIND_OBJECT:
        target = intent["target"]
        if not is_supported_label(target):
            return unsupported_object_response(target)
        try:
            vision_result = normalize_find_result(detect_object(target), target)
        except Exception:
            LOGGER.exception("vision detect_object failed")
            vision_result = {"object": target, "detected": False,
                             "error": "VISION_ERROR"}
    elif name == DESCRIBE_SCENE:
        try:
            vision_result = normalize_scene_result(describe_scene())
        except Exception:
            LOGGER.exception("vision describe_scene failed")
            vision_result = {"error": "VISION_ERROR"}

    return build_response(intent, vision_result)


def handle_command(
    text: str,
    detect_object: Optional[Callable[[str], Dict]] = None,
    describe_scene: Optional[Callable[[], List[Dict]]] = None,
    on_stop: Optional[Callable[[], None]] = None,
) -> str:
    """Turn recognized text into the response string to speak (one-shot).

    detect_object / describe_scene default to the mock vision functions.
    on_stop is called when the user says stop (e.g. to cut off speech).
    """
    if detect_object is None or describe_scene is None:
        mock = MockVision()
        detect_object = detect_object or mock.detect_object
        describe_scene = describe_scene or mock.describe_scene
    return respond_to_intent(parse_intent(text), detect_object, describe_scene, on_stop)


DEMO_COMMANDS = [
    "Where is my water bottle?",
    "Can you tell me what objects are in front of me?",
    "Where is my guitar?",
    "Stop.",
    "What's the weather today?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="SAAM controller demo (mock vision)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="type commands instead of running the demo list")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show which parser produced each intent")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(name)s: %(message)s")

    def show(text: str) -> None:
        print("You said : {}".format(text))
        print("Intent   : {}".format(parse_intent(text)))
        print("Response : {}\n".format(handle_command(text)))

    if args.interactive:
        print("Type a command (empty line to quit).")
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                break
            show(text)
    else:
        for text in DEMO_COMMANDS:
            show(text)


if __name__ == "__main__":
    main()
