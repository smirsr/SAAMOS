"""Shared test helpers (no hardware, no real model).

SlmDisabledTestCase turns the SLM off for the duration of each test only, so
no test file changes the environment for the others.

make_fake_llama() builds a stand-in "llama.cpp" command out of a small Python
script, so the real subprocess code path can be tested on Windows and Linux.
"""

import contextlib
import logging
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Warnings the code logs on purpose (timeouts, bad settings) would otherwise
# clutter the test output.
logging.getLogger("saam").addHandler(logging.NullHandler())


class SlmDisabledTestCase(unittest.TestCase):
    """Base class: SAAM_DISABLE_SLM=1 during each test, restored afterwards."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"SAAM_DISABLE_SLM": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)


@contextlib.contextmanager
def slm_environment(binary=None, model=None, **extra):
    """Temporarily point the SLM engine at a (fake) binary and model.

    Everything set here is undone when the block exits.
    """
    env = {}
    if binary is not None:
        env["SAAM_LLAMA_BIN"] = binary
    if model is not None:
        env["SAAM_MODEL_PATH"] = model
    env.update(extra)
    with mock.patch.dict(os.environ, env):
        if "SAAM_DISABLE_SLM" not in extra:
            os.environ.pop("SAAM_DISABLE_SLM", None)
        yield


def make_fake_model(directory):
    """Create an empty stand-in model file and return its path."""
    path = os.path.join(directory, "fake_model.gguf")
    with open(path, "w") as handle:
        handle.write("not a real model")
    return path


def make_fake_llama(directory, body):
    """Create a fake llama.cpp command running the Python code in `body`.

    The command accepts any arguments (the real flags are passed to it) and
    they are available to the script as sys.argv. Returns the command's path.
    """
    script = os.path.join(directory, "fake_llama_impl.py")
    with open(script, "w") as handle:
        handle.write(body)
    if os.name == "nt":
        launcher = os.path.join(directory, "fake_llama.bat")
        with open(launcher, "w") as handle:
            handle.write('@echo off\r\n"{}" "{}" %*\r\n'.format(sys.executable, script))
    else:
        launcher = os.path.join(directory, "fake_llama")
        with open(launcher, "w") as handle:
            handle.write('#!/bin/sh\nexec "{}" "{}" "$@"\n'.format(sys.executable, script))
        os.chmod(launcher, 0o755)
    return launcher
