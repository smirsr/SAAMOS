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
from saam.core.intent_parser import parse_intent, rule_based_parse  # noqa: E402
from tests.helpers import SlmDisabledTestCase  # noqa: E402


def find(target):
    return {"intent": "FIND_OBJECT", "target": target}


SCENE = {"intent": "DESCRIBE_SCENE", "target": None}
STOP = {"intent": "STOP", "target": None}
UNKNOWN = {"intent": "UNKNOWN", "target": None}


class RuleBasedTests(SlmDisabledTestCase):
    def check(self, text, expected):
        self.assertEqual(rule_based_parse(text), expected, text)

    def test_find_object(self):
        self.check("Where is my water bottle?", find("bottle"))
        self.check("where's my cup", find("cup"))
        self.check("Can you find my phone?", find("cell phone"))
        self.check("Find the laptop please", find("laptop"))
        self.check("Is there a chair nearby?", find("chair"))
        self.check("Do you see a bottle?", find("bottle"))
        self.check("Where did I put my keys", find("keys"))
        self.check("Help me find a mug", find("cup"))
        self.check("Where is the stop sign?", find("stop sign"))
        self.check("I lost my remote", find("remote"))

    def test_describe_scene(self):
        self.check("Can you tell me what objects are in front of me?", SCENE)
        self.check("What's around me?", SCENE)
        self.check("What do you see?", SCENE)
        self.check("Describe the room", SCENE)
        self.check("What is in front of me", SCENE)
        self.check("Is there anything in front of me?", SCENE)

    def test_stop(self):
        self.check("Stop.", STOP)
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


class SlmFallbackTests(SlmDisabledTestCase):
    """parse_intent() must use the SLM when it is good, and the rules otherwise."""

    def setUp(self):
        super().setUp()
        original = slm_engine.slm_parse
        self.addCleanup(setattr, slm_engine, "slm_parse", original)

    def test_valid_slm_output_is_used(self):
        slm_engine.slm_parse = lambda text: {"intent": "FIND_OBJECT", "target": "water bottle"}
        self.assertEqual(parse_intent("anything"), find("bottle"))

    def test_slm_unavailable_uses_rules(self):
        slm_engine.slm_parse = lambda text: None
        self.assertEqual(parse_intent("Stop."), STOP)

    def test_slm_invalid_output_uses_rules(self):
        slm_engine.slm_parse = lambda text: {"intent": "DANCE", "target": None}
        self.assertEqual(parse_intent("Stop."), STOP)
        slm_engine.slm_parse = lambda text: {"intent": "FIND_OBJECT", "target": None}
        self.assertEqual(parse_intent("Where is my cup?"), find("cup"))

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
