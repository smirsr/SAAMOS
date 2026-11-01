"""Standalone tests for the response builder and controller (mock vision only).

Run from the repository root:
    python -m unittest tests.test_controller -v
or:
    python tests/test_controller.py
"""

import os
import sys
import unittest

os.environ["SAAM_DISABLE_SLM"] = "1"  # tests never call the real model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saam.core.controller import MockVision, handle_command  # noqa: E402
from saam.core.response_builder import build_response  # noqa: E402


class ControllerTests(unittest.TestCase):
    def test_find_object_found(self):
        self.assertEqual(handle_command("Where is my water bottle?"),
                         "Your bottle is on the left.")

    def test_find_object_center_and_right(self):
        self.assertEqual(handle_command("Where is my laptop?"),
                         "Your laptop is in front of you.")
        self.assertEqual(handle_command("Where is my cup?"),
                         "Your cup is on the right.")

    def test_find_object_not_found(self):
        self.assertEqual(handle_command("Where is my guitar?"),
                         "I couldn't find your guitar.")

    def test_low_confidence_counts_as_not_found(self):
        vision = MockVision([{"object": "bottle", "detected": True,
                              "position": "left", "confidence": 0.3}])
        self.assertEqual(
            handle_command("Where is my bottle?", detect_object=vision.detect_object,
                           describe_scene=vision.describe_scene),
            "I couldn't find your bottle.")

    def test_describe_scene(self):
        self.assertEqual(
            handle_command("What objects are in front of me?"),
            "I can see a bottle on the left, a cup on the right, and a laptop in front of you.")

    def test_describe_scene_is_limited_to_three_objects(self):
        scene = [
            {"object": name, "detected": True, "position": "left", "confidence": conf}
            for name, conf in [("bottle", 0.9), ("cup", 0.8), ("chair", 0.7), ("book", 0.6)]
        ]
        vision = MockVision(scene)
        reply = handle_command("What do you see?", detect_object=vision.detect_object,
                               describe_scene=vision.describe_scene)
        self.assertNotIn("book", reply)
        self.assertIn("a chair", reply)

    def test_describe_empty_scene(self):
        vision = MockVision([])
        self.assertEqual(
            handle_command("What do you see?", detect_object=vision.detect_object,
                           describe_scene=vision.describe_scene),
            "I don't see any objects right now.")

    def test_stop_calls_callback_and_skips_vision(self):
        calls = []

        def vision_must_not_run(*args):
            raise AssertionError("vision should not be called for STOP")

        reply = handle_command("Stop.", detect_object=vision_must_not_run,
                               describe_scene=vision_must_not_run,
                               on_stop=lambda: calls.append("stopped"))
        self.assertEqual(reply, "Okay, stopping.")
        self.assertEqual(calls, ["stopped"])

    def test_unknown_and_empty_input(self):
        expected = "Sorry, I didn't catch that. Could you repeat it?"
        self.assertEqual(handle_command("What's the weather today?"), expected)
        self.assertEqual(handle_command(""), expected)

    def test_vision_error_dict_gives_camera_message(self):
        def broken(target):
            return {"object": target, "detected": False, "error": "CAMERA_UNAVAILABLE"}

        self.assertEqual(handle_command("Where is my cup?", detect_object=broken),
                         "Sorry, I can't use the camera right now.")

    def test_vision_exception_never_crashes(self):
        def broken(target):
            raise RuntimeError("camera exploded")

        def broken_scene():
            raise RuntimeError("camera exploded")

        self.assertEqual(handle_command("Where is my cup?", detect_object=broken,
                                        describe_scene=broken_scene),
                         "Sorry, I can't use the camera right now.")
        self.assertEqual(handle_command("What do you see?", detect_object=broken,
                                        describe_scene=broken_scene),
                         "I don't see any objects right now.")


class ResponseBuilderTests(unittest.TestCase):
    def test_odd_input_never_raises(self):
        unknown = "Sorry, I didn't catch that. Could you repeat it?"
        self.assertEqual(build_response(None), unknown)
        self.assertEqual(build_response({"intent": "FIND_OBJECT", "target": None}), unknown)
        self.assertEqual(build_response({"intent": "DESCRIBE_SCENE", "target": None}, "oops"),
                         "I don't see any objects right now.")

    def test_unknown_position(self):
        detection = {"object": "cup", "detected": True, "position": None, "confidence": 0.9}
        self.assertEqual(build_response({"intent": "FIND_OBJECT", "target": "cup"}, detection),
                         "I can see your cup, but I'm not sure where it is.")


if __name__ == "__main__":
    unittest.main()
