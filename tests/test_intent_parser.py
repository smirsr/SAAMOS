"""Standalone tests for saam/core/intent_parser.py (no hardware, no model).

Run from the repository root:
    python -m unittest tests.test_intent_parser -v
or:
    python tests/test_intent_parser.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saam.core import slm_engine  # noqa: E402
from saam.core.intent_parser import (  # noqa: E402
    parse_intent, rule_based_parse, singularize)
from tests.helpers import SlmDisabledTestCase  # noqa: E402


def intent(name, target=None):
    return {"intent": name, "target": target}


def find(target):
    return intent("FIND_OBJECT", target)


def start_find(target):
    return intent("START_FIND", target)


def count(target):
    return intent("COUNT_OBJECTS", target)


SCENE = intent("DESCRIBE_SCENE")
STOP = intent("STOP")
UNKNOWN = intent("UNKNOWN")


class RuleBasedTests(SlmDisabledTestCase):
    def check(self, text, expected):
        self.assertEqual(rule_based_parse(text), expected, text)

    # --- the original commands (unchanged) ---
    def test_original_examples(self):
        self.check("Where is my water bottle?", find("bottle"))
        self.check("Can you tell me what objects are in front of me?", SCENE)
        self.check("Stop.", STOP)

    def test_one_shot_find_questions(self):
        self.check("where's my cup", find("cup"))
        self.check("Is there a chair nearby?", find("chair"))
        self.check("Do you see a bottle?", find("bottle"))
        self.check("Where did I put my keys", find("keys"))
        self.check("Where is the stop sign?", find("stop sign"))
        self.check("I lost my remote", find("remote"))

    def test_describe_scene(self):
        self.check("What's around me?", SCENE)
        self.check("What do you see?", SCENE)
        self.check("Describe the room", SCENE)
        self.check("What is in front of me", SCENE)
        self.check("Is there anything in front of me?", SCENE)

    def test_stop(self):
        self.check("stop talking", STOP)
        self.check("Cancel", STOP)
        self.check("Never mind", STOP)
        self.check("be quiet", STOP)

    def test_unknown(self):
        self.check("What's the weather today?", UNKNOWN)
        self.check("Hello there", UNKNOWN)
        self.check("", UNKNOWN)
        self.check("   ", UNKNOWN)
        self.check("Where is the thing I left on the table yesterday", UNKNOWN)

    # --- new commands for operating modes ---
    def test_find_mode_commands(self):
        self.check("Find my bottle", start_find("bottle"))
        self.check("Find the laptop please", start_find("laptop"))
        self.check("Can you find my phone?", start_find("cell phone"))
        self.check("Help me find a mug", start_find("cup"))
        self.check("Look for my water bottle", start_find("bottle"))
        self.check("Find the stop sign", start_find("stop sign"))

    def test_stop_searching(self):
        for text in ["Stop searching", "stop looking", "Cancel the search", "end search"]:
            self.check(text, intent("STOP_FIND"))

    def test_scan_commands(self):
        for text in ["Scan my surroundings", "Start scanning", "scan the room"]:
            self.check(text, intent("START_SCAN"))
        for text in ["Stop scanning", "cancel the scan"]:
            self.check(text, intent("STOP_SCAN"))

    def test_count_commands(self):
        self.check("How many bottles do you see?", count("bottle"))
        self.check("How many cups are there?", count("cup"))
        self.check("how many water bottles do you see", count("bottle"))
        self.check("How many people are in front of me?", count("person"))
        self.check("How many glasses can you see", count("glass"))
        self.check("How many bottles", count("bottle"))
        # no single label to count: answer as a scene
        self.check("How many objects do you see?", SCENE)
        # nothing to count
        self.check("How many do you see?", UNKNOWN)

    def test_read_and_repeat(self):
        for text in ["Read this text", "read that sign", "What does this say?"]:
            self.check(text, intent("READ_TEXT"))
        for text in ["Repeat that", "repeat", "Say that again", "What did you say?"]:
            self.check(text, intent("REPEAT"))

    def test_stop_variants_are_not_confused(self):
        # "stop" alone ends everything; "stop searching/scanning" are specific.
        self.check("Stop", STOP)
        self.check("Stop searching", intent("STOP_FIND"))
        self.check("Stop scanning", intent("STOP_SCAN"))

    def test_singularize(self):
        for plural, single in [("bottles", "bottle"), ("cups", "cup"), ("people", "person"),
                               ("glasses", "glass"), ("boxes", "box"), ("berries", "berry"),
                               ("bus", "bus"), ("glass", "glass"), ("scissors", "scissors")]:
            self.assertEqual(singularize(plural), single, plural)


class SlmFallbackTests(SlmDisabledTestCase):
    """parse_intent() must use the SLM when it is good, and the rules otherwise."""

    def setUp(self):
        super().setUp()
        original = slm_engine.slm_parse
        self.addCleanup(setattr, slm_engine, "slm_parse", original)

    def test_valid_slm_output_is_used(self):
        slm_engine.slm_parse = lambda text: {"intent": "FIND_OBJECT", "target": "water bottle"}
        self.assertEqual(parse_intent("anything"), find("bottle"))

    def test_new_intents_from_the_slm_are_accepted(self):
        slm_engine.slm_parse = lambda text: {"intent": "START_FIND", "target": "mug"}
        self.assertEqual(parse_intent("anything"), start_find("cup"))
        slm_engine.slm_parse = lambda text: {"intent": "REPEAT", "target": "ignored"}
        self.assertEqual(parse_intent("anything"), intent("REPEAT"))

    def test_slm_unavailable_uses_rules(self):
        slm_engine.slm_parse = lambda text: None
        self.assertEqual(parse_intent("Stop."), STOP)

    def test_slm_invalid_output_uses_rules(self):
        slm_engine.slm_parse = lambda text: {"intent": "DANCE", "target": None}
        self.assertEqual(parse_intent("Stop."), STOP)
        slm_engine.slm_parse = lambda text: {"intent": "FIND_OBJECT", "target": None}
        self.assertEqual(parse_intent("Where is my cup?"), find("cup"))
        slm_engine.slm_parse = lambda text: {"intent": "COUNT_OBJECTS", "target": None}
        self.assertEqual(parse_intent("How many cups do you see?"), count("cup"))

    def test_slm_exception_uses_rules(self):
        def boom(text):
            raise RuntimeError("model crashed")
        slm_engine.slm_parse = boom
        self.assertEqual(parse_intent("What do you see?"), SCENE)

    def test_slm_unknown_still_tries_rules(self):
        slm_engine.slm_parse = lambda text: {"intent": "UNKNOWN", "target": None}
        self.assertEqual(parse_intent("Stop."), STOP)

    def test_bad_input_never_raises(self):
        self.assertEqual(parse_intent(None), UNKNOWN)
        self.assertEqual(parse_intent(""), UNKNOWN)


if __name__ == "__main__":
    unittest.main()
