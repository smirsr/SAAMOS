"""Standalone tests for saam/core/slm_engine.py.

The subprocess tests use a fake "llama.cpp" command (a tiny Python script), so
they exercise the real subprocess code, including a real timeout, without a
model or any hardware.

Run from the repository root:
    python -m unittest tests.test_slm_engine -v
or:
    python tests/test_slm_engine.py
"""

import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saam.core import slm_engine  # noqa: E402
from saam.core.intent_parser import parse_intent  # noqa: E402
from tests.helpers import (  # noqa: E402
    make_fake_llama, make_fake_model, slm_environment)

STOP = {"intent": "STOP", "target": None}

# Fake model: checks it was given the prompt file, then answers.
OK_BODY = '''
import json, os, sys
args = sys.argv[1:]
prompt_file = args[args.index("-f") + 1]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "seen_prompt.txt"), "w") as record:
    record.write(prompt_file)
with open(prompt_file, encoding="utf-8") as handle:
    prompt = handle.read()
if "Command: Where is my cup?" in prompt:
    print('JSON: ' + json.dumps({"intent": "FIND_OBJECT", "target": "cup"}))
'''

# Fake model that is far too slow.
SLOW_BODY = '''
import time
time.sleep(8)
print('{"intent": "STOP", "target": null}')
'''

# Fake model that crashes.
FAIL_BODY = '''
import sys
sys.stderr.write("error: unknown argument\\n")
sys.exit(2)
'''

# Fake model that answers with nonsense.
GARBAGE_BODY = '''
print("I am a language model and I would love to help you with that.")
'''


class TimeoutSettingTests(unittest.TestCase):
    def get(self, value):
        with mock.patch.dict(os.environ, {"SAAM_SLM_TIMEOUT": value}):
            return slm_engine.get_timeout()

    def test_default_when_unset(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("SAAM_SLM_TIMEOUT", None)
            self.assertEqual(slm_engine.get_timeout(), slm_engine.DEFAULT_TIMEOUT_S)

    def test_valid_values_are_used(self):
        self.assertEqual(self.get("2.5"), 2.5)
        self.assertEqual(self.get("10"), 10.0)

    def test_invalid_values_use_default(self):
        for bad in ["abc", "", "   ", "0", "-5", "nan", "inf", "-inf", "1e999x"]:
            self.assertEqual(self.get(bad), slm_engine.DEFAULT_TIMEOUT_S, repr(bad))

    def test_huge_value_is_capped(self):
        self.assertEqual(self.get("100000"), slm_engine.MAX_TIMEOUT_S)

    def test_bad_extra_args_use_default(self):
        with mock.patch.dict(os.environ, {"SAAM_LLAMA_EXTRA_ARGS": '--flag "unclosed'}):
            self.assertEqual(slm_engine.get_extra_args(), slm_engine.DEFAULT_EXTRA_ARGS)

    def test_extra_args_can_be_overridden(self):
        with mock.patch.dict(os.environ, {"SAAM_LLAMA_EXTRA_ARGS": "--foo 1 --bar"}):
            self.assertEqual(slm_engine.get_extra_args(), ["--foo", "1", "--bar"])


class ExtractJsonTests(unittest.TestCase):
    def test_takes_last_object(self):
        output = 'JSON: {"intent": "STOP", "target": null}\nJSON: {"intent": "FIND_OBJECT", "target": "cup"}'
        self.assertEqual(slm_engine.extract_json(output),
                         {"intent": "FIND_OBJECT", "target": "cup"})

    def test_handles_garbage(self):
        self.assertIsNone(slm_engine.extract_json("no json here"))
        self.assertIsNone(slm_engine.extract_json("{not valid}"))
        self.assertIsNone(slm_engine.extract_json(""))


class AvailabilityTests(unittest.TestCase):
    def test_unavailable_without_config(self):
        with mock.patch.dict(os.environ):
            for name in ("SAAM_LLAMA_BIN", "SAAM_MODEL_PATH", "SAAM_DISABLE_SLM"):
                os.environ.pop(name, None)
            self.assertFalse(slm_engine.is_available())
            self.assertIsNone(slm_engine.slm_parse("Stop."))

    def test_disable_flag_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = make_fake_llama(folder, OK_BODY)
            model = make_fake_model(folder)
            with slm_environment(binary, model, SAAM_DISABLE_SLM="1"):
                self.assertFalse(slm_engine.is_available())


class SubprocessTests(unittest.TestCase):
    """Runs the real subprocess code against a fake llama.cpp command."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.model = make_fake_model(self.folder)

    def run_with(self, body, **extra):
        binary = make_fake_llama(self.folder, body)
        return slm_environment(binary, self.model, **extra)

    def test_successful_run_returns_parsed_json(self):
        with self.run_with(OK_BODY, SAAM_SLM_TIMEOUT="20"):
            self.assertEqual(slm_engine.slm_parse("Where is my cup?"),
                             {"intent": "FIND_OBJECT", "target": "cup"})

    def test_prompt_file_is_removed_afterwards(self):
        with self.run_with(OK_BODY, SAAM_SLM_TIMEOUT="20"):
            slm_engine.slm_parse("Where is my cup?")
        with open(os.path.join(self.folder, "seen_prompt.txt")) as record:
            prompt_file = record.read()
        self.assertFalse(os.path.exists(prompt_file))

    def test_real_timeout_falls_back_to_rules(self):
        with self.run_with(SLOW_BODY, SAAM_SLM_TIMEOUT="1"):
            started = time.time()
            self.assertIsNone(slm_engine.slm_parse("Stop."))
            # Must return soon after the 1 s timeout, not wait for the 8 s sleep
            # (on Windows the fake command is a .bat wrapper with a child process).
            self.assertLess(time.time() - started, 5)
            # parse_intent still answers, using the rule-based parser.
            self.assertEqual(parse_intent("Stop."), STOP)

    def test_invalid_timeout_setting_does_not_break_a_run(self):
        with self.run_with(OK_BODY, SAAM_SLM_TIMEOUT="not-a-number"):
            self.assertEqual(slm_engine.slm_parse("Where is my cup?"),
                             {"intent": "FIND_OBJECT", "target": "cup"})

    def test_crashing_model_falls_back_to_rules(self):
        with self.run_with(FAIL_BODY, SAAM_SLM_TIMEOUT="20"):
            self.assertIsNone(slm_engine.slm_parse("Stop."))
            self.assertEqual(parse_intent("Stop."), STOP)

    def test_garbage_output_falls_back_to_rules(self):
        with self.run_with(GARBAGE_BODY, SAAM_SLM_TIMEOUT="20"):
            self.assertIsNone(slm_engine.slm_parse("Stop."))
            self.assertEqual(parse_intent("Stop."), STOP)


if __name__ == "__main__":
    unittest.main()
