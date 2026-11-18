"""Standalone tests for the response builder, controller and object labels.

Uses mock vision only: no hardware, no model.

Run from the repository root:
    python -m unittest tests.test_controller -v
or:
    python tests/test_controller.py
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saam.core import object_labels  # noqa: E402
from saam.core.controller import MockVision, handle_command  # noqa: E402
from saam.core.response_builder import build_response  # noqa: E402
from tests.helpers import SlmDisabledTestCase  # noqa: E402

UNKNOWN_REPLY = "Sorry, I didn't catch that. Could you repeat it?"
CAMERA_REPLY = "Sorry, I can't use the camera right now."


class ControllerTests(SlmDisabledTestCase):
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

    def test_describe_empty_scene_means_camera_worked(self):
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

    def test_stop_still_answers_if_callback_fails(self):
        def broken():
            raise RuntimeError("audio module crashed")

        self.assertEqual(handle_command("Stop.", on_stop=broken), "Okay, stopping.")

    def test_unknown_and_empty_input(self):
        self.assertEqual(handle_command("What's the weather today?"), UNKNOWN_REPLY)
        self.assertEqual(handle_command(""), UNKNOWN_REPLY)


class CameraErrorTests(SlmDisabledTestCase):
    """A vision failure must never sound like "nothing found" or "empty room"."""

    def ask(self, text, detect=None, describe=None):
        mock_vision = MockVision()
        return handle_command(text,
                              detect_object=detect or mock_vision.detect_object,
                              describe_scene=describe or mock_vision.describe_scene)

    def test_find_error_dict(self):
        def broken(target):
            return {"object": target, "detected": False, "error": "CAMERA_UNAVAILABLE"}
        self.assertEqual(self.ask("Where is my cup?", detect=broken), CAMERA_REPLY)

    def test_find_exception(self):
        def broken(target):
            raise RuntimeError("camera exploded")
        self.assertEqual(self.ask("Where is my cup?", detect=broken), CAMERA_REPLY)

    def test_find_none_means_not_in_view_not_a_camera_error(self):
        # The agreed vision format returns None when the object is simply not there.
        self.assertEqual(self.ask("Where is my cup?", detect=lambda target: None),
                         "I couldn't find your cup.")

    def test_find_garbage_return_value_is_a_camera_error(self):
        self.assertEqual(self.ask("Where is my cup?", detect=lambda target: "garbage"),
                         CAMERA_REPLY)
        self.assertEqual(self.ask("Where is my cup?", detect=lambda target: 42),
                         CAMERA_REPLY)

    def test_describe_error_dict(self):
        self.assertEqual(
            self.ask("What do you see?",
                     describe=lambda: {"error": "CAMERA_UNAVAILABLE"}),
            CAMERA_REPLY)

    def test_describe_exception(self):
        def broken():
            raise RuntimeError("camera exploded")
        self.assertEqual(self.ask("What do you see?", describe=broken), CAMERA_REPLY)

    def test_describe_bad_return_value(self):
        self.assertEqual(self.ask("What do you see?", describe=lambda: None),
                         CAMERA_REPLY)


class ResponseBuilderTests(unittest.TestCase):
    def test_odd_input_never_raises(self):
        self.assertEqual(build_response(None), UNKNOWN_REPLY)
        self.assertEqual(build_response({"intent": "FIND_OBJECT", "target": None}),
                         UNKNOWN_REPLY)

    def test_missing_vision_result_is_a_camera_problem(self):
        self.assertEqual(build_response({"intent": "DESCRIBE_SCENE", "target": None}, "oops"),
                         CAMERA_REPLY)
        self.assertEqual(build_response({"intent": "FIND_OBJECT", "target": "cup"}, None),
                         CAMERA_REPLY)

    def test_unknown_position(self):
        detection = {"object": "cup", "detected": True, "position": None, "confidence": 0.9}
        self.assertEqual(build_response({"intent": "FIND_OBJECT", "target": "cup"}, detection),
                         "I can see your cup, but I'm not sure where it is.")


class ObjectLabelTests(SlmDisabledTestCase):
    """Only objects the vision module supports are sent to vision."""

    def labels_file(self, text):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = os.path.join(folder.name, "labels.txt")
        with open(path, "w") as handle:
            handle.write(text)
        patcher = mock.patch.dict(os.environ, {"SAAM_LABELS_FILE": path})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_loader_ignores_comments_blank_lines_and_case(self):
        self.labels_file("# comment\n\nBottle\n  cup  \n")
        self.assertEqual(object_labels.load_supported_labels(), {"bottle", "cup"})

    def test_empty_list_means_no_restriction(self):
        self.labels_file("# nothing yet\n")
        self.assertIsNone(object_labels.load_supported_labels())
        self.assertTrue(object_labels.is_supported_label("guitar"))

    def test_unreadable_file_means_no_restriction(self):
        with mock.patch.dict(os.environ, {"SAAM_LABELS_FILE": "/no/such/file.txt"}):
            self.assertIsNone(object_labels.load_supported_labels())

    def test_unsupported_object_is_not_sent_to_vision(self):
        self.labels_file("bottle\ncup\n")

        def vision_must_not_run(target):
            raise AssertionError("vision should not be asked about unsupported labels")

        self.assertEqual(
            handle_command("Where is my guitar?", detect_object=vision_must_not_run),
            "Sorry, I can't look for a guitar yet.")

    def test_article_matches_the_word(self):
        self.labels_file("bottle\n")
        self.assertEqual(handle_command("Where is my umbrella?"),
                         "Sorry, I can't look for an umbrella yet.")

    def test_supported_object_still_works(self):
        self.labels_file("bottle\ncup\n")
        self.assertEqual(handle_command("Where is my water bottle?"),
                         "Your bottle is on the left.")


if __name__ == "__main__":
    unittest.main()
