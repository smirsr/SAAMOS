"""Tests for the agreed detection format:

    {"label": "bottle", "confidence": 0.85, "position": "left", "bbox": [x1, y1, x2, y2]}

and for the "not found" vs "failed" rules. No hardware needed.

Run from the repository root:
    python -m unittest tests.test_detections -v
or:
    python tests/test_detections.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saam.core.controller import MOCK_SCENE, MockVision, handle_command  # noqa: E402
from saam.core.detections import (  # noqa: E402
    label_of, normalize_find_result, normalize_scene_result)
from saam.core.modes import Mode, ModeController, ModeSettings  # noqa: E402
from tests.helpers import SlmDisabledTestCase  # noqa: E402

CAMERA = "Sorry, I can't use the camera right now."


def det(label, position="left", confidence=0.85, bbox=None):
    """A detection in the agreed format."""
    return {"label": label, "confidence": confidence, "position": position,
            "bbox": bbox or [10, 20, 110, 220]}


class NormalizeFindTests(unittest.TestCase):
    def test_agreed_format_is_accepted_and_bbox_is_kept(self):
        result = normalize_find_result(det("bottle"), "bottle")
        self.assertTrue(result["detected"])
        self.assertEqual(result["label"], "bottle")
        self.assertEqual(result["object"], "bottle")
        self.assertEqual(result["position"], "left")
        self.assertEqual(result["bbox"], [10, 20, 110, 220])

    def test_old_format_is_still_accepted(self):
        old = {"object": "bottle", "detected": True, "position": "right", "confidence": 0.9}
        result = normalize_find_result(old, "bottle")
        self.assertTrue(result["detected"])
        self.assertEqual(result["label"], "bottle")

    def test_old_format_can_still_say_not_detected(self):
        old = {"object": "bottle", "detected": False, "position": None, "confidence": 0.0}
        self.assertFalse(normalize_find_result(old, "bottle")["detected"])

    def test_nothing_returned_means_not_found(self):
        for nothing in (None, {}, []):
            result = normalize_find_result(nothing, "bottle")
            self.assertFalse(result["detected"], repr(nothing))
            self.assertNotIn("error", result)

    def test_error_is_passed_through_and_is_not_not_found(self):
        error = {"error": "CAMERA_UNAVAILABLE"}
        self.assertEqual(normalize_find_result(error, "bottle"), error)

    def test_garbage_is_an_error(self):
        for garbage in ("oops", 42, True):
            self.assertTrue(normalize_find_result(garbage, "bottle").get("error"), repr(garbage))

    def test_a_detection_of_a_different_object_is_not_an_answer(self):
        self.assertFalse(normalize_find_result(det("cup"), "bottle")["detected"])

    def test_label_match_ignores_case(self):
        self.assertTrue(normalize_find_result(det("Bottle"), "bottle")["detected"])

    def test_a_list_picks_the_most_confident_match(self):
        result = normalize_find_result(
            [det("bottle", "left", 0.4), det("cup", "right", 0.99), det("bottle", "center", 0.8)],
            "bottle")
        self.assertEqual((result["position"], result["confidence"]), ("center", 0.8))
        self.assertFalse(normalize_find_result([det("cup")], "bottle")["detected"])

    def test_label_of_accepts_either_key(self):
        self.assertEqual(label_of({"label": "cup"}), "cup")
        self.assertEqual(label_of({"object": "cup"}), "cup")
        self.assertIsNone(label_of({}))
        self.assertIsNone(label_of("cup"))


class NormalizeSceneTests(unittest.TestCase):
    def test_agreed_format_list(self):
        result = normalize_scene_result([det("bottle"), det("cup", "right")])
        self.assertEqual([d["label"] for d in result], ["bottle", "cup"])
        self.assertTrue(all(d["detected"] for d in result))
        self.assertEqual(result[0]["bbox"], [10, 20, 110, 220])

    def test_empty_list_means_nothing_seen_not_an_error(self):
        self.assertEqual(normalize_scene_result([]), [])

    def test_error_dict_none_and_garbage_are_errors(self):
        self.assertEqual(normalize_scene_result({"error": "CAMERA_UNAVAILABLE"}),
                         {"error": "CAMERA_UNAVAILABLE"})
        for bad in (None, "oops", 42, {}):
            self.assertTrue(normalize_scene_result(bad).get("error"), repr(bad))

    def test_junk_entries_are_skipped(self):
        self.assertEqual([d["label"] for d in normalize_scene_result([det("cup"), None, "x"])],
                         ["cup"])


class MockVisionFormatTests(unittest.TestCase):
    def test_mock_uses_the_agreed_format(self):
        for item in MOCK_SCENE:
            self.assertEqual(set(item), {"label", "confidence", "position", "bbox"})
            self.assertEqual(len(item["bbox"]), 4)

    def test_mock_returns_none_for_an_object_that_is_not_there(self):
        vision = MockVision()
        self.assertIsNone(vision.detect_object("guitar"))
        self.assertEqual(vision.detect_object("bottle")["position"], "left")

    def test_added_objects_have_a_label_and_box(self):
        vision = MockVision()
        vision.add("book", "right")
        item = vision.detect_object("book")
        self.assertEqual((item["label"], item["position"]), ("book", "right"))
        self.assertEqual(len(item["bbox"]), 4)


class OneShotWithAgreedFormatTests(SlmDisabledTestCase):
    """The original commands against a vision module that uses the agreed format."""

    def ask(self, text, detect=None, describe=None):
        return handle_command(text, detect_object=detect or (lambda t: None),
                              describe_scene=describe or (lambda: []))

    def test_found(self):
        self.assertEqual(self.ask("Where is my water bottle?", detect=lambda t: det("bottle", "left")),
                         "Your bottle is on the left.")
        self.assertEqual(self.ask("Where is my cup?", detect=lambda t: det("cup", "center")),
                         "Your cup is in front of you.")

    def test_not_in_view(self):
        self.assertEqual(self.ask("Where is my bottle?"), "I couldn't find your bottle.")

    def test_low_confidence_is_not_found(self):
        self.assertEqual(self.ask("Where is my bottle?", detect=lambda t: det("bottle", "left", 0.3)),
                         "I couldn't find your bottle.")

    def test_wrong_object_is_not_announced(self):
        self.assertEqual(self.ask("Where is my bottle?", detect=lambda t: det("cup", "left")),
                         "I couldn't find your bottle.")

    def test_error_is_a_camera_problem(self):
        self.assertEqual(self.ask("Where is my bottle?", detect=lambda t: {"error": "CAMERA_UNAVAILABLE"}),
                         CAMERA)

    def test_scene(self):
        scene = [det("bottle", "left", 0.9), det("cup", "right", 0.8), det("laptop", "center", 0.7)]
        self.assertEqual(self.ask("What do you see?", describe=lambda: scene),
                         "I can see a bottle on the left, a cup on the right, and a laptop in front of you.")

    def test_empty_scene_vs_error(self):
        self.assertEqual(self.ask("What do you see?", describe=lambda: []),
                         "I don't see any objects right now.")
        self.assertEqual(self.ask("What do you see?", describe=lambda: {"error": "CAMERA_UNAVAILABLE"}),
                         CAMERA)


class ModesWithAgreedFormatTests(SlmDisabledTestCase):
    def make(self, vision):
        spoken = []
        clock = [0.0]
        controller = ModeController(vision=vision, speak=spoken.append,
                                    settings=ModeSettings(confirm_ticks=2),
                                    clock=lambda: clock[0])
        return controller, spoken

    def test_find_mode_uses_label_position_and_none_for_not_found(self):
        vision = MockVision([])
        controller, spoken = self.make(vision)
        controller.handle_text("Find my bottle")
        self.assertEqual([controller.tick(), controller.tick()], [None, None])   # not in view: quiet
        vision.add("bottle", "left")
        self.assertEqual([controller.tick(), controller.tick()], [None, "Your bottle is on the left."])
        vision.set_position("bottle", "right")
        self.assertEqual([controller.tick(), controller.tick()], [None, "Your bottle is now on the right."])
        vision.remove("bottle")
        self.assertEqual([controller.tick(), controller.tick()], [None, "I lost sight of your bottle."])
        self.assertIs(controller.mode, Mode.FINDING)

    def test_scan_mode_uses_labels(self):
        vision = MockVision()
        controller, spoken = self.make(vision)
        controller.handle_text("Scan my surroundings")
        self.assertIn("a bottle on the left", controller.tick())
        vision.add("book", "left")
        self.assertEqual([controller.tick(), controller.tick()], [None, "A book appeared on the left."])
        vision.remove("cup")
        self.assertEqual([controller.tick(), controller.tick()], [None, "The cup is gone."])

    def test_counting_from_the_scene_uses_labels(self):
        vision = MockVision([det("bottle", "left"), det("bottle", "right"), det("cup", "center")])
        controller, spoken = self.make(vision)
        self.assertEqual(controller.handle_text("How many bottles do you see?"), "I can see 2 bottles.")
        self.assertEqual(controller.handle_text("How many cups do you see?"), "I can see one cup.")
        self.assertEqual(controller.handle_text("How many people do you see?"),
                         "I don't see any people.")

    def test_count_function_may_use_the_label_key(self):
        class Counting(MockVision):
            def count_objects(self, target):
                return {"label": target, "count": 3}

        controller, spoken = self.make(Counting())
        self.assertEqual(controller.handle_text("How many cups do you see?"), "I can see 3 cups.")

    def test_error_dict_from_detect_object_ends_find_mode_after_three(self):
        vision = MockVision()
        controller, spoken = self.make(vision)
        controller.handle_text("Find my bottle")
        vision.detect_object = lambda target: {"error": "CAMERA_UNAVAILABLE"}
        self.assertEqual([controller.tick() for _ in range(3)],
                         [None, None, "Sorry, I can't use the camera right now. I've stopped."])
        self.assertIs(controller.mode, Mode.IDLE)

    def test_none_from_detect_object_never_stops_find_mode(self):
        vision = MockVision([])
        controller, spoken = self.make(vision)
        controller.handle_text("Find my bottle")
        self.assertEqual([controller.tick() for _ in range(20)], [None] * 20)
        self.assertIs(controller.mode, Mode.FINDING)


if __name__ == "__main__":
    unittest.main()
