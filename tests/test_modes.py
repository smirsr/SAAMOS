"""Standalone tests for operating modes, push-to-talk and the new commands.

No TI board, camera, microphone, speaker or GPIO needed: vision is mocked, time
is a fake clock, and "buttons" are plain ButtonEvent values.

Run from the repository root:
    python -m unittest tests.test_modes -v
or:
    python tests/test_modes.py
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saam.core import response_builder as rb  # noqa: E402
from saam.core.controller import MockVision, handle_command  # noqa: E402
from saam.core.modes import (  # noqa: E402
    ButtonEvent, Mode, ModeController, ModeSettings, parse_button_event)
from tests.helpers import SlmDisabledTestCase  # noqa: E402

CAMERA = "Sorry, I can't use the camera right now."
CAMERA_STOPPED = "Sorry, I can't use the camera right now. I've stopped."


class FakeClock:
    """Time that only moves when a test says so, so tests never really wait."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Harness:
    """A ModeController wired to mock vision, a fake clock and recording callbacks."""

    def __init__(self, scene=None, vision=None, **settings):
        self.clock = FakeClock()
        self.spoken = []
        self.calls = []
        self.vision = vision if vision is not None else MockVision(scene)
        options = {"confirm_ticks": 2}
        options.update(settings)
        self.controller = ModeController(
            vision=self.vision,
            speak=self.spoken.append,
            stop_speaking=lambda: self.calls.append("stop_speaking"),
            start_listening=lambda: self.calls.append("start_listening"),
            stop_listening=lambda: self.calls.append("stop_listening"),
            settings=ModeSettings(**options),
            clock=self.clock,
        )

    def say(self, text):
        return self.controller.handle_text(text)

    def ticks(self, count=1):
        return [self.controller.tick() for _ in range(count)]


def with_labels(test, text):
    """Make the vision label list contain exactly these labels for one test."""
    folder = tempfile.TemporaryDirectory()
    test.addCleanup(folder.cleanup)
    path = os.path.join(folder.name, "labels.txt")
    with open(path, "w") as handle:
        handle.write(text)
    patcher = mock.patch.dict(os.environ, {"SAAM_LABELS_FILE": path})
    patcher.start()
    test.addCleanup(patcher.stop)


class ModeSwitchingTests(SlmDisabledTestCase):
    def test_starts_idle(self):
        self.assertIs(Harness().controller.mode, Mode.IDLE)

    def test_short_press_starts_listening(self):
        h = Harness()
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        self.assertIs(h.controller.mode, Mode.LISTENING)
        self.assertEqual(h.calls, ["start_listening"])

    def test_command_goes_listening_speaking_idle(self):
        h = Harness()
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        h.say("Where is my water bottle?")
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertEqual(h.controller.mode_history(),
                         [Mode.IDLE, Mode.LISTENING, Mode.SPEAKING, Mode.IDLE])
        self.assertEqual(h.spoken, ["Your bottle is on the left."])

    def test_listening_times_out_by_itself(self):
        h = Harness()
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        h.clock.advance(5)
        self.assertIsNone(h.controller.tick())
        self.assertIs(h.controller.mode, Mode.LISTENING)
        h.clock.advance(6)
        self.assertEqual(h.controller.tick(), "I didn't hear a command.")
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_second_short_press_while_listening_is_ignored(self):
        h = Harness()
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        self.assertEqual(h.calls, ["start_listening"])

    def test_short_press_interrupts_speech(self):
        h = Harness(speech_blocks=False)
        h.say("Where is my water bottle?")
        self.assertIs(h.controller.mode, Mode.SPEAKING)
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        self.assertIs(h.controller.mode, Mode.LISTENING)
        self.assertEqual(h.calls, ["stop_speaking", "start_listening"])

    def test_listening_during_find_returns_to_find(self):
        h = Harness()
        h.say("Find my bottle")
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        self.assertIs(h.controller.mode, Mode.LISTENING)
        h.say("What do you see?")
        self.assertIs(h.controller.mode, Mode.FINDING)

    def test_listening_failure_recovers(self):
        h = Harness()

        def broken():
            raise RuntimeError("microphone missing")

        h.controller._start_listening_cb = broken
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertEqual(h.spoken, ["Sorry, I can't listen right now."])

    def test_empty_recognition_returns_to_idle(self):
        h = Harness()
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        self.assertEqual(h.say(""), rb.UNKNOWN_MESSAGE)
        self.assertIs(h.controller.mode, Mode.IDLE)


class FindModeTests(SlmDisabledTestCase):
    def test_find_my_bottle_starts_find_mode(self):
        h = Harness()
        self.assertEqual(h.say("Find my bottle"), "Okay, I'll look for your bottle.")
        self.assertIs(h.controller.mode, Mode.FINDING)
        self.assertIs(h.controller.activity, Mode.FINDING)

    def test_announces_once_when_found(self):
        h = Harness()
        h.say("Find my bottle")
        self.assertEqual(h.ticks(5), [None, "Your bottle is on the left.", None, None, None])

    def test_reports_position_changes(self):
        h = Harness()
        h.say("Find my bottle")
        h.ticks(2)
        h.vision.set_position("bottle", "right")
        self.assertEqual(h.ticks(2), [None, "Your bottle is now on the right."])
        h.vision.set_position("bottle", "center")
        self.assertEqual(h.ticks(2), [None, "Your bottle is now in front of you."])

    def test_reports_lost_and_found_again(self):
        h = Harness()
        h.say("Find my bottle")
        h.ticks(2)
        h.vision.remove("bottle")
        self.assertEqual(h.ticks(2), [None, "I lost sight of your bottle."])
        h.vision.add("bottle", "center")
        self.assertEqual(h.ticks(2), [None, "Your bottle is in front of you."])

    def test_one_frame_flicker_is_not_announced(self):
        h = Harness()
        h.say("Find my bottle")
        h.ticks(2)
        h.vision.remove("bottle")
        self.assertIsNone(h.controller.tick())
        h.vision.add("bottle", "left")
        self.assertEqual(h.ticks(3), [None, None, None])

    def test_missing_object_stays_quiet_then_gives_up(self):
        h = Harness()
        h.say("Find my guitar")
        self.assertEqual(h.ticks(10), [None] * 10)
        self.assertIs(h.controller.mode, Mode.FINDING)
        h.clock.advance(121)
        self.assertEqual(h.controller.tick(),
                         "I couldn't find your guitar. I've stopped searching.")
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertIsNone(h.controller.activity)

    def test_timeout_message_after_object_was_seen(self):
        h = Harness()
        h.say("Find my bottle")
        h.ticks(2)
        h.clock.advance(121)
        self.assertEqual(h.controller.tick(), "I've stopped searching for your bottle.")

    def test_stop_searching_by_voice(self):
        h = Harness()
        h.say("Find my bottle")
        self.assertEqual(h.say("Stop searching"), "Okay, I've stopped searching.")
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertEqual(h.ticks(3), [None, None, None])

    def test_stop_searching_when_not_searching(self):
        h = Harness()
        self.assertEqual(h.say("Stop searching"), "I'm not searching for anything right now.")
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_stop_ends_find_mode_and_cuts_speech(self):
        h = Harness()
        h.say("Find my bottle")
        self.assertEqual(h.say("Stop."), "Okay, stopping.")
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertIsNone(h.controller.activity)
        self.assertIn("stop_speaking", h.calls)

    def test_new_find_replaces_the_old_one(self):
        h = Harness()
        h.say("Find my bottle")
        h.say("Find my cup")
        self.assertEqual(h.ticks(2), [None, "Your cup is on the right."])

    def test_one_shot_questions_still_work_during_find(self):
        h = Harness()
        h.say("Find my bottle")
        self.assertEqual(h.say("Where is my cup?"), "Your cup is on the right.")
        self.assertIs(h.controller.mode, Mode.FINDING)

    def test_works_with_other_objects_not_just_bottle(self):
        scene = [{"object": "cell phone", "detected": True, "position": "right", "confidence": 0.8},
                 {"object": "umbrella", "detected": True, "position": "left", "confidence": 0.7}]
        for phrase, label, expected in [("Find my phone", "cell phone", "Your cell phone is on the right."),
                                        ("Find the umbrella", "umbrella", "Your umbrella is on the left.")]:
            h = Harness(scene=scene)
            self.assertEqual(h.say(phrase), "Okay, I'll look for your {}.".format(label))
            self.assertEqual(h.ticks(2), [None, expected])

    def test_unsupported_object_does_not_start_find_mode(self):
        with_labels(self, "bottle\ncup\n")
        h = Harness()
        self.assertEqual(h.say("Find my guitar"), "Sorry, I can't look for a guitar yet.")
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertIsNone(h.controller.activity)

    def test_supported_object_starts_find_mode_when_labels_are_set(self):
        with_labels(self, "bottle\ncup\n")
        h = Harness()
        self.assertEqual(h.say("Find my water bottle"), "Okay, I'll look for your bottle.")
        self.assertIs(h.controller.mode, Mode.FINDING)

    def test_repeated_camera_errors_end_find_mode(self):
        h = Harness()
        h.say("Find my bottle")
        h.vision.detect_object = lambda target: {"error": "CAMERA_UNAVAILABLE"}
        self.assertEqual(h.ticks(3), [None, None, CAMERA_STOPPED])
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_a_single_camera_error_is_ignored(self):
        h = Harness()
        h.say("Find my bottle")
        good = h.vision.detect_object
        h.vision.detect_object = lambda target: {"error": "CAMERA_UNAVAILABLE"}
        self.assertEqual(h.ticks(2), [None, None])
        h.vision.detect_object = good
        self.assertEqual(h.ticks(2), [None, "Your bottle is on the left."])
        self.assertIs(h.controller.mode, Mode.FINDING)

    def test_vision_exception_is_a_camera_error(self):
        h = Harness()
        h.say("Find my bottle")

        def broken(target):
            raise RuntimeError("camera exploded")

        h.vision.detect_object = broken
        self.assertEqual(h.ticks(3), [None, None, CAMERA_STOPPED])


class ScanModeTests(SlmDisabledTestCase):
    def test_scan_describes_the_scene_then_stays_quiet(self):
        h = Harness()
        self.assertEqual(h.say("Scan my surroundings"),
                         "Okay, I'll scan your surroundings and tell you what changes.")
        self.assertIs(h.controller.mode, Mode.SCANNING)
        self.assertEqual(h.ticks(4), [
            "I can see a bottle on the left, a cup on the right, and a laptop in front of you.",
            None, None, None])

    def test_reports_a_new_object(self):
        h = Harness()
        h.say("Scan my surroundings")
        h.ticks(1)
        h.vision.add("book", "left")
        self.assertEqual(h.ticks(2), [None, "A book appeared on the left."])

    def test_reports_an_object_that_left(self):
        h = Harness()
        h.say("Scan my surroundings")
        h.ticks(1)
        h.vision.remove("cup")
        self.assertEqual(h.ticks(2), [None, "The cup is gone."])

    def test_reports_a_count_change(self):
        h = Harness()
        h.say("Scan my surroundings")
        h.ticks(1)
        h.vision.add("bottle", "right")
        self.assertEqual(h.ticks(2), [None, "Now I see 2 bottles."])

    def test_ignores_position_only_changes(self):
        h = Harness()
        h.say("Scan my surroundings")
        h.ticks(1)
        h.vision.set_position("bottle", "right")
        self.assertEqual(h.ticks(4), [None] * 4)

    def test_one_frame_flicker_is_not_announced(self):
        h = Harness()
        h.say("Scan my surroundings")
        h.ticks(1)
        h.vision.add("book", "left")
        self.assertIsNone(h.controller.tick())
        h.vision.remove("book")
        self.assertEqual(h.ticks(3), [None] * 3)

    def test_empty_scene_is_reported_as_empty(self):
        h = Harness(scene=[])
        h.say("Scan my surroundings")
        self.assertEqual(h.ticks(1), ["I don't see any objects right now."])

    def test_stop_scanning(self):
        h = Harness()
        h.say("Scan my surroundings")
        self.assertEqual(h.say("Stop scanning"), "Okay, I've stopped scanning.")
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertEqual(h.ticks(3), [None] * 3)

    def test_stop_scanning_when_not_scanning(self):
        h = Harness()
        self.assertEqual(h.say("Stop scanning"), "I'm not scanning right now.")

    def test_stop_searching_does_not_end_a_scan(self):
        h = Harness()
        h.say("Scan my surroundings")
        self.assertEqual(h.say("Stop searching"), "I'm not searching for anything right now.")
        self.assertIs(h.controller.mode, Mode.SCANNING)

    def test_scan_gives_up_by_itself(self):
        h = Harness()
        h.say("Scan my surroundings")
        h.ticks(1)
        h.clock.advance(301)
        self.assertEqual(h.controller.tick(), rb.SCAN_TIMEOUT_MESSAGE)
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_repeated_camera_errors_end_scan_mode(self):
        h = Harness()
        h.say("Scan my surroundings")
        h.vision.describe_scene = lambda: {"error": "CAMERA_UNAVAILABLE"}
        self.assertEqual(h.ticks(3), [None, None, CAMERA_STOPPED])
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_scan_replaces_find(self):
        h = Harness()
        h.say("Find my bottle")
        h.say("Scan my surroundings")
        self.assertIs(h.controller.activity, Mode.SCANNING)


class CountTests(SlmDisabledTestCase):
    def two(self, label):
        return [{"object": label, "detected": True, "position": pos, "confidence": 0.9}
                for pos in ("left", "right")]

    def test_counts_several(self):
        h = Harness(scene=self.two("bottle"))
        self.assertEqual(h.say("How many bottles do you see?"), "I can see 2 bottles.")

    def test_counts_one(self):
        self.assertEqual(Harness().say("How many bottles do you see?"), "I can see one bottle.")

    def test_counts_zero(self):
        self.assertEqual(Harness().say("How many guitars do you see?"),
                         "I don't see any guitars.")

    def test_works_for_any_object_and_plural(self):
        self.assertEqual(Harness(scene=self.two("person")).say("How many people are there?"),
                         "I can see 2 people.")
        self.assertEqual(Harness(scene=self.two("glass")).say("How many glasses can you see"),
                         "I can see 2 glasses.")
        self.assertEqual(Harness().say("How many cups do you see?"), "I can see one cup.")

    def test_low_confidence_detections_are_not_counted(self):
        scene = [{"object": "bottle", "detected": True, "position": "left", "confidence": 0.2},
                 {"object": "bottle", "detected": True, "position": "right", "confidence": 0.9}]
        self.assertEqual(Harness(scene=scene).say("How many bottles do you see?"),
                         "I can see one bottle.")

    def test_uses_the_vision_count_function_when_it_exists(self):
        class CountingVision(MockVision):
            def count_objects(self, target):
                return {"object": target, "count": 5}

            def describe_scene(self):
                raise AssertionError("describe_scene should not be used")

        h = Harness(vision=CountingVision())
        self.assertEqual(h.say("How many cups do you see?"), "I can see 5 cups.")

    def test_camera_error_is_not_reported_as_zero(self):
        h = Harness()
        h.vision.describe_scene = lambda: {"error": "CAMERA_UNAVAILABLE"}
        self.assertEqual(h.say("How many bottles do you see?"), CAMERA)

    def test_unsupported_label(self):
        with_labels(self, "bottle\n")
        self.assertEqual(Harness().say("How many guitars do you see?"),
                         "Sorry, I can't look for a guitar yet.")

    def test_how_many_objects_is_a_scene_description(self):
        self.assertIn("I can see a bottle", Harness().say("How many objects do you see?"))


class RepeatTests(SlmDisabledTestCase):
    def test_repeat_that(self):
        h = Harness()
        h.say("Where is my water bottle?")
        self.assertEqual(h.say("Repeat that"), "Your bottle is on the left.")
        self.assertEqual(h.spoken, ["Your bottle is on the left."] * 2)

    def test_repeat_with_nothing_said_yet(self):
        h = Harness()
        self.assertEqual(h.say("Repeat that"), "I haven't said anything yet.")

    def test_repeat_does_not_replace_the_last_response(self):
        h = Harness()
        h.say("Where is my water bottle?")
        h.say("Repeat that")
        h.say("Repeat that")
        self.assertEqual(h.controller.last_response, "Your bottle is on the left.")
        self.assertEqual(h.spoken.count("Your bottle is on the left."), 3)

    def test_repeat_button_event(self):
        h = Harness()
        h.say("Where is my cup?")
        self.assertEqual(h.controller.on_button_event(ButtonEvent.REPEAT),
                         "Your cup is on the right.")
        self.assertEqual(h.spoken[-1], "Your cup is on the right.")

    def test_repeat_button_with_nothing_said(self):
        h = Harness()
        self.assertEqual(h.controller.on_button_event(ButtonEvent.REPEAT),
                         "I haven't said anything yet.")

    def test_repeats_announcements_made_by_find_mode(self):
        h = Harness()
        h.say("Find my bottle")
        h.ticks(2)
        self.assertEqual(h.say("Repeat that"), "Your bottle is on the left.")

    def test_repeat_cuts_off_current_speech_first(self):
        h = Harness(speech_blocks=False)
        h.say("Where is my cup?")
        h.controller.on_button_event(ButtonEvent.REPEAT)
        self.assertIn("stop_speaking", h.calls)


class ButtonTests(SlmDisabledTestCase):
    def test_long_press_stops_find_mode(self):
        h = Harness()
        h.say("Find my bottle")
        self.assertEqual(h.controller.on_button_event(ButtonEvent.LONG_PRESS),
                         "Okay, I've stopped searching.")
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertEqual(h.ticks(2), [None, None])

    def test_long_press_stops_scan_mode(self):
        h = Harness()
        h.say("Scan my surroundings")
        self.assertEqual(h.controller.on_button_event(ButtonEvent.LONG_PRESS),
                         "Okay, I've stopped scanning.")
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_long_press_when_idle_is_silent(self):
        h = Harness()
        self.assertIsNone(h.controller.on_button_event(ButtonEvent.LONG_PRESS))
        self.assertEqual(h.spoken, [])
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_long_press_cancels_listening(self):
        h = Harness()
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        h.controller.on_button_event(ButtonEvent.LONG_PRESS)
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertIn("stop_listening", h.calls)

    def test_long_press_cuts_off_speech(self):
        h = Harness(speech_blocks=False)
        h.say("Where is my cup?")
        h.controller.on_button_event(ButtonEvent.LONG_PRESS)
        self.assertIs(h.controller.mode, Mode.IDLE)
        self.assertIn("stop_speaking", h.calls)

    def test_keyboard_keys_work_like_the_button(self):
        h = Harness()
        h.controller.on_key("s")
        self.assertIs(h.controller.mode, Mode.LISTENING)
        h.controller.on_key("l")
        self.assertIs(h.controller.mode, Mode.IDLE)
        h.controller.on_key(" ")
        self.assertIs(h.controller.mode, Mode.LISTENING)
        h.say("Where is my cup?")
        self.assertEqual(h.controller.on_key("r"), "Your cup is on the right.")

    def test_events_can_be_given_by_name(self):
        for value, expected in [("short_press", ButtonEvent.SHORT_PRESS),
                                ("LONG_PRESS", ButtonEvent.LONG_PRESS),
                                ("repeat", ButtonEvent.REPEAT),
                                ("double_press", ButtonEvent.REPEAT),
                                (ButtonEvent.LONG_PRESS, ButtonEvent.LONG_PRESS)]:
            self.assertIs(parse_button_event(value), expected, value)

    def test_unknown_events_are_ignored(self):
        h = Harness()
        for junk in ["banana", "", None, 42, "x"]:
            self.assertIsNone(h.controller.on_button_event(junk))
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_a_typical_session_of_simulated_presses(self):
        h = Harness()
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        h.say("Find my bottle")
        self.assertIs(h.controller.mode, Mode.FINDING)
        h.ticks(2)
        h.controller.on_button_event(ButtonEvent.REPEAT)
        self.assertEqual(h.spoken[-1], "Your bottle is on the left.")
        h.controller.on_button_event(ButtonEvent.LONG_PRESS)
        self.assertIs(h.controller.mode, Mode.IDLE)


class ReadTextTests(SlmDisabledTestCase):
    def test_placeholder_until_ocr_is_installed(self):
        h = Harness()
        self.assertEqual(h.say("Read this text"), "Reading text isn't set up yet.")
        self.assertIn(Mode.READING, h.controller.mode_history())
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_reads_text_found_by_ocr(self):
        h = Harness()
        h.controller._read_text_cb = lambda: {"text": "EXIT\n  Door  "}
        self.assertEqual(h.say("Read this text"), "It says: EXIT Door")

    def test_no_text_found(self):
        h = Harness()
        h.controller._read_text_cb = lambda: {"text": ""}
        self.assertEqual(h.say("Read this text"), "I couldn't find any text.")

    def test_ocr_error_and_exception(self):
        h = Harness()
        h.controller._read_text_cb = lambda: {"error": "OCR_FAILED"}
        self.assertEqual(h.say("Read this text"), "Sorry, I can't read text right now.")

        def broken():
            raise RuntimeError("ocr crashed")

        h.controller._read_text_cb = broken
        self.assertEqual(h.say("Read this text"), "Sorry, I can't read text right now.")
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_long_text_is_shortened(self):
        h = Harness()
        h.controller._read_text_cb = lambda: {"text": "word " * 500}
        self.assertLessEqual(len(h.say("Read this text")), len("It says: ") + rb.MAX_READ_CHARS)

    def test_stop_during_ocr_keeps_quiet(self):
        h = Harness()

        def ocr_that_gets_stopped():
            self.assertIs(h.controller.mode, Mode.READING)
            h.controller.stop()                      # the user pressed stop mid-read
            return {"text": "this must not be spoken"}

        h.controller._read_text_cb = ocr_that_gets_stopped
        self.assertEqual(h.say("Read this text"), "")
        self.assertEqual(h.spoken, [])
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_reading_returns_to_find_mode(self):
        h = Harness()
        h.say("Find my bottle")
        h.say("Read this text")
        self.assertIs(h.controller.mode, Mode.FINDING)


class StopSafetyTests(SlmDisabledTestCase):
    """"Stop" must always work, and no mode may get stuck."""

    def in_each_mode(self):
        """Yield (name, harness) with the controller put into each mode."""
        h = Harness()
        yield "IDLE", h
        h = Harness()
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        yield "LISTENING", h
        h = Harness()
        h.say("Find my bottle")
        yield "FINDING", h
        h = Harness()
        h.say("Scan my surroundings")
        yield "SCANNING", h
        h = Harness(speech_blocks=False)
        h.say("Where is my cup?")
        yield "SPEAKING", h

    def test_stop_returns_to_idle_from_every_mode(self):
        for name, h in self.in_each_mode():
            h.controller.stop()
            self.assertIs(h.controller.mode, Mode.IDLE, name)
            self.assertIsNone(h.controller.activity, name)

    def test_spoken_stop_returns_to_idle_from_every_mode(self):
        for name, h in self.in_each_mode():
            self.assertEqual(h.say("Stop"), "Okay, stopping.", name)
            self.assertIsNone(h.controller.activity, name)
            # (when speech does not block, "Okay, stopping." is still playing)
            h.controller.speech_finished()
            self.assertIs(h.controller.mode, Mode.IDLE, name)

    def test_button_stop_returns_to_idle_from_every_mode(self):
        for name, h in self.in_each_mode():
            h.controller.on_button_event(ButtonEvent.LONG_PRESS)
            self.assertIs(h.controller.mode, Mode.IDLE, name)

    def test_stop_during_reading(self):
        h = Harness()
        seen = []

        def ocr():
            seen.append(h.controller.mode)
            h.controller.stop()
            seen.append(h.controller.mode)
            return {"text": "ignored"}

        h.controller._read_text_cb = ocr
        h.say("Read this text")
        self.assertEqual(seen, [Mode.READING, Mode.IDLE])

    def test_stop_works_even_if_audio_callbacks_fail(self):
        h = Harness()

        def broken():
            raise RuntimeError("audio crashed")

        h.controller._stop_speaking_cb = broken
        h.controller._stop_listening_cb = broken
        h.controller.on_button_event(ButtonEvent.SHORT_PRESS)
        h.controller.stop()
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_speaking_failure_does_not_get_stuck(self):
        h = Harness()

        def broken(text):
            raise RuntimeError("speaker unplugged")

        h.controller._speak_cb = broken
        self.assertEqual(h.say("Where is my cup?"), "Your cup is on the right.")
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_stop_twice_is_harmless(self):
        h = Harness()
        h.say("Find my bottle")
        h.controller.stop()
        h.controller.stop()
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_find_mode_cannot_run_forever(self):
        h = Harness()
        h.say("Find my guitar")
        for step in range(100000):
            h.clock.advance(0.5)
            h.controller.tick()
            if h.controller.activity is None:
                break
        self.assertIsNone(h.controller.activity)
        self.assertLess(step, 400)           # 120 s at two ticks per second is about 240

    def test_scan_mode_cannot_run_forever(self):
        h = Harness()
        h.say("Scan my surroundings")
        for step in range(100000):
            h.clock.advance(0.5)
            h.controller.tick()
            if h.controller.activity is None:
                break
        self.assertIsNone(h.controller.activity)
        self.assertLess(step, 800)           # 300 s at two ticks per second is about 600

    def test_speaking_that_never_reports_finished_times_out(self):
        h = Harness(speech_blocks=False)
        h.say("Where is my cup?")
        self.assertIs(h.controller.mode, Mode.SPEAKING)
        self.assertIsNone(h.controller.tick())
        self.assertIs(h.controller.mode, Mode.SPEAKING)
        h.clock.advance(40)
        h.controller.tick()
        self.assertIs(h.controller.mode, Mode.IDLE)

    def test_speech_finished_returns_to_idle_or_the_running_mode(self):
        h = Harness(speech_blocks=False)
        h.say("Where is my cup?")
        h.controller.speech_finished()
        self.assertIs(h.controller.mode, Mode.IDLE)
        h.say("Find my bottle")
        self.assertIs(h.controller.mode, Mode.SPEAKING)
        h.controller.speech_finished()
        self.assertIs(h.controller.mode, Mode.FINDING)

    def test_no_announcements_while_still_speaking(self):
        h = Harness(speech_blocks=False)
        h.say("Find my bottle")
        self.assertEqual(h.ticks(5), [None] * 5)       # busy speaking: stays quiet
        h.controller.speech_finished()
        self.assertEqual(h.ticks(2), [None, "Your bottle is on the left."])

    def test_stop_is_not_blocked_by_speech_that_is_still_playing(self):
        """A press on another thread must work while speak() is blocked."""
        h = Harness()
        started, release = threading.Event(), threading.Event()

        def blocking_speak(text):
            started.set()
            release.wait(5)

        h.controller._speak_cb = blocking_speak
        worker = threading.Thread(target=h.say, args=("Where is my cup?",))
        worker.start()
        try:
            self.assertTrue(started.wait(5))
            began = time.time()
            h.controller.stop()
            self.assertLess(time.time() - began, 1.0)
            self.assertIs(h.controller.mode, Mode.IDLE)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIs(h.controller.mode, Mode.IDLE)      # the finished speech did not undo the stop

    def test_stop_is_not_blocked_by_a_slow_camera(self):
        h = Harness()
        h.say("Scan my surroundings")
        started, release = threading.Event(), threading.Event()
        results = []

        def slow_scene():
            started.set()
            release.wait(5)
            return MockVision().describe_scene()

        h.vision.describe_scene = slow_scene
        worker = threading.Thread(target=lambda: results.append(h.controller.tick()))
        worker.start()
        try:
            self.assertTrue(started.wait(5))
            began = time.time()
            h.controller.stop()
            self.assertLess(time.time() - began, 1.0)
        finally:
            release.set()
            worker.join(5)
        self.assertEqual(results, [None])                 # the late camera result was discarded
        self.assertEqual(h.spoken, ["Okay, I'll scan your surroundings and tell you what changes."])
        self.assertIs(h.controller.mode, Mode.IDLE)


class OriginalCommandsTests(SlmDisabledTestCase):
    """The original one-shot commands behave exactly as before."""

    COMMANDS = [
        "Where is my water bottle?",
        "Can you tell me what objects are in front of me?",
        "Where is my guitar?",
        "Stop.",
        "What's the weather today?",
    ]

    def test_mode_controller_gives_the_same_answers_as_handle_command(self):
        for text in self.COMMANDS:
            self.assertEqual(Harness().say(text), handle_command(text), text)

    def test_original_answers(self):
        h = Harness()
        self.assertEqual(h.say("Where is my water bottle?"), "Your bottle is on the left.")
        self.assertEqual(h.say("Can you tell me what objects are in front of me?"),
                         "I can see a bottle on the left, a cup on the right, and a laptop in front of you.")
        self.assertEqual(h.say("Where is my guitar?"), "I couldn't find your guitar.")
        self.assertEqual(h.say("Stop."), "Okay, stopping.")
        self.assertEqual(h.say("What's the weather today?"), rb.UNKNOWN_MESSAGE)

    def test_handle_command_still_works_without_modes(self):
        self.assertEqual(handle_command("Where is my water bottle?"), "Your bottle is on the left.")
        # "Find ..." is answered once, because there is no mode controller here.
        self.assertEqual(handle_command("Find my bottle"), "Your bottle is on the left.")
        self.assertEqual(handle_command("Stop searching"), "Okay, stopping.")
        # Commands that need modes are not understood by the old one-shot path.
        self.assertEqual(handle_command("Scan my surroundings"), rb.UNKNOWN_MESSAGE)

    def test_camera_error_wording_is_unchanged(self):
        h = Harness()
        h.vision.describe_scene = lambda: {"error": "CAMERA_UNAVAILABLE"}
        self.assertEqual(h.say("What do you see?"), CAMERA)


if __name__ == "__main__":
    unittest.main()
