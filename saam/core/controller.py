"""Main controller: recognized text -> intent -> vision (if needed) -> response.

handle_command(text) is the entry point described in docs/integration-guide.md.
Until the real vision module is ready, it uses MockVision so the controller can
be run and tested on its own. The prototype starts with rule-based parsing and
mock vision; the SLM is optional (see slm_engine.py).

Run the demo from the repository root:

    python -m saam.core.controller
    python -m saam.core.controller --interactive

How STOP reaches the audio side
-------------------------------
The controller does not own the speaker or the microphone. When the user says
stop, handle_command() calls the on_stop callback it was given. The audio
owner passes in a function (proposed name: stop_speaking) that cuts off any
speech that is playing. See "STOP and the audio loop" in the integration guide.
"""

import argparse
import logging
from typing import Callable, Dict, List, Optional

from .intent_parser import DESCRIBE_SCENE, FIND_OBJECT, STOP, parse_intent
from .object_labels import is_supported_label
from .response_builder import build_response, unsupported_object_response

LOGGER = logging.getLogger(__name__)

# Simulated vision results (same format as the real vision module).
MOCK_SCENE = [
    {"object": "bottle", "detected": True, "position": "left", "confidence": 0.91},
    {"object": "cup", "detected": True, "position": "right", "confidence": 0.84},
    {"object": "laptop", "detected": True, "position": "center", "confidence": 0.77},
]


class MockVision:
    """Stand-in for the vision module, with the same two functions."""

    def __init__(self, scene: Optional[List[Dict]] = None):
        self.scene = list(MOCK_SCENE if scene is None else scene)

    def detect_object(self, target: str) -> Dict:
        for item in self.scene:
            if item.get("object") == target:
                return dict(item)
        return {"object": target, "detected": False, "position": None,
                "confidence": 0.0}

    def describe_scene(self) -> List[Dict]:
        return [dict(item) for item in self.scene]


def handle_command(
    text: str,
    detect_object: Optional[Callable[[str], Dict]] = None,
    describe_scene: Optional[Callable[[], List[Dict]]] = None,
    on_stop: Optional[Callable[[], None]] = None,
) -> str:
    """Turn recognized text into the response string to speak.

    detect_object / describe_scene default to the mock vision functions.
    on_stop is called when the user says stop (e.g. to cut off speech).
    Never raises: a failing module produces a spoken error message instead.
    A vision failure (exception, error dict, or bad return value) is reported
    as a camera problem, never as "nothing found" or an empty scene.
    """
    if detect_object is None or describe_scene is None:
        mock = MockVision()
        detect_object = detect_object or mock.detect_object
        describe_scene = describe_scene or mock.describe_scene

    intent = parse_intent(text)
    name = intent["intent"]
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
            vision_result = detect_object(target)
        except Exception:
            LOGGER.exception("vision detect_object failed")
            vision_result = {"object": target, "detected": False,
                             "error": "VISION_ERROR"}
    elif name == DESCRIBE_SCENE:
        try:
            vision_result = describe_scene()
        except Exception:
            LOGGER.exception("vision describe_scene failed")
            vision_result = {"error": "VISION_ERROR"}

    return build_response(intent, vision_result)


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
