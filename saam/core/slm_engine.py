"""Small language model (SLM) engine, run through llama.cpp.

This module hides how the model is run. The rest of the system only calls
slm_parse(), which returns a dict like {"intent": "STOP", "target": None} or
None when the model is unavailable, too slow, or gives unusable output. The
caller (intent_parser.py) then falls back to the rule-based parser.

The SLM is optional. Nothing here is hardcoded to one machine; it is switched
on with environment variables:

    SAAM_LLAMA_BIN         llama.cpp command or path, e.g. llama-cli
    SAAM_MODEL_PATH        path to the .gguf model file
    SAAM_SLM_TIMEOUT       seconds to wait before giving up (default 10,
                           must be a number above 0, capped at 60)
    SAAM_LLAMA_EXTRA_ARGS  extra llama.cpp flags (default: --no-display-prompt)
    SAAM_DISABLE_SLM       set to 1 to skip the model entirely

Invalid values never cause errors. They are logged and the default is used.

The prompt is passed to llama.cpp as a temporary file (-f) rather than on the
command line, which avoids quoting and length problems. llama.cpp flag names
change between versions: check them with `llama-cli --help` on the board and
adjust SAAM_LLAMA_EXTRA_ARGS if needed.

Everything runs locally. No network or cloud API is used.
"""

import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)

ENV_BIN = "SAAM_LLAMA_BIN"
ENV_MODEL = "SAAM_MODEL_PATH"
ENV_TIMEOUT = "SAAM_SLM_TIMEOUT"
ENV_EXTRA_ARGS = "SAAM_LLAMA_EXTRA_ARGS"
ENV_DISABLE = "SAAM_DISABLE_SLM"

DEFAULT_TIMEOUT_S = 10.0
MAX_TIMEOUT_S = 60.0
DEFAULT_EXTRA_ARGS = ["--no-display-prompt"]
MAX_NEW_TOKENS = 48

# Kept short on purpose: prompt processing is slow on the board's CPU, so every
# extra example costs time. The rule-based parser remains the default.
PROMPT_TEMPLATE = """You turn a spoken command into JSON.
Intents: FIND_OBJECT (answer once about an object), START_FIND (keep looking for an object), STOP_FIND, START_SCAN, STOP_SCAN, COUNT_OBJECTS (how many of an object), DESCRIBE_SCENE, READ_TEXT, REPEAT, STOP, UNKNOWN.
Reply with the JSON only.

Command: Where is my water bottle?
JSON: {"intent": "FIND_OBJECT", "target": "bottle"}
Command: Find my bottle
JSON: {"intent": "START_FIND", "target": "bottle"}
Command: How many cups do you see?
JSON: {"intent": "COUNT_OBJECTS", "target": "cup"}
Command: What objects are in front of me?
JSON: {"intent": "DESCRIBE_SCENE", "target": null}
Command: Stop.
JSON: {"intent": "STOP", "target": null}
Command: What is the weather today?
JSON: {"intent": "UNKNOWN", "target": null}
Command: <<TEXT>>
JSON:"""


def _binary() -> Optional[str]:
    name = os.environ.get(ENV_BIN)
    if not name:
        return None
    return shutil.which(name)


def _model_path() -> Optional[str]:
    path = os.environ.get(ENV_MODEL)
    if path and os.path.isfile(path):
        return path
    return None


def is_available() -> bool:
    """True if the model is switched on and both the binary and model exist."""
    if os.environ.get(ENV_DISABLE) == "1":
        return False
    return _binary() is not None and _model_path() is not None


def get_timeout() -> float:
    """Timeout in seconds from SAAM_SLM_TIMEOUT. Invalid values use the default."""
    raw = os.environ.get(ENV_TIMEOUT)
    if raw is None:
        return DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        LOGGER.warning("%s=%r is not a number; using %.0f s",
                       ENV_TIMEOUT, raw, DEFAULT_TIMEOUT_S)
        return DEFAULT_TIMEOUT_S
    if not math.isfinite(value) or value <= 0:
        LOGGER.warning("%s=%r must be a number above 0; using %.0f s",
                       ENV_TIMEOUT, raw, DEFAULT_TIMEOUT_S)
        return DEFAULT_TIMEOUT_S
    if value > MAX_TIMEOUT_S:
        LOGGER.warning("%s=%r is too large; using %.0f s",
                       ENV_TIMEOUT, raw, MAX_TIMEOUT_S)
        return MAX_TIMEOUT_S
    return value


def get_extra_args() -> List[str]:
    """Extra llama.cpp flags from SAAM_LLAMA_EXTRA_ARGS (bad quoting uses the default)."""
    raw = os.environ.get(ENV_EXTRA_ARGS)
    if raw is None:
        return list(DEFAULT_EXTRA_ARGS)
    try:
        return shlex.split(raw)
    except ValueError:
        LOGGER.warning("%s=%r could not be parsed; using defaults", ENV_EXTRA_ARGS, raw)
        return list(DEFAULT_EXTRA_ARGS)


def build_prompt(text: str) -> str:
    """Insert the command into the few-shot prompt."""
    return PROMPT_TEMPLATE.replace("<<TEXT>>", text.strip())


def extract_json(output: str) -> Optional[Dict]:
    """Return the last flat JSON object found in the model output, or None."""
    matches = re.findall(r"\{[^{}]*\}", output or "")
    if not matches:
        return None
    try:
        parsed = json.loads(matches[-1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _write_prompt_file(prompt: str) -> Optional[str]:
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="saam_prompt_",
            delete=False, encoding="utf-8")
        with handle:
            handle.write(prompt)
        return handle.name
    except OSError as error:
        LOGGER.warning("SLM could not write its prompt file (%s); using fallback", error)
        return None


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _run_model(command: List[str], timeout: float):
    """Run the command and return (returncode, stdout, stderr).

    Output goes to temporary files, not pipes. If the command starts helper
    processes (a wrapper script, for example), a timeout then returns as soon
    as the main process is killed. With pipes, Windows would keep waiting until
    every helper process had exited.
    """
    with tempfile.TemporaryFile() as out_file, tempfile.TemporaryFile() as err_file:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,  # llama.cpp can never wait for interactive input
            stdout=out_file,
            stderr=err_file,
            timeout=timeout,
        )
        out_file.seek(0)
        err_file.seek(0)
        return (completed.returncode,
                out_file.read().decode("utf-8", errors="replace"),
                err_file.read().decode("utf-8", errors="replace"))


def slm_parse(text: str) -> Optional[Dict]:
    """Ask the model for an intent. Returns a dict, or None on any problem."""
    if not is_available():
        return None

    timeout = get_timeout()
    prompt_path = _write_prompt_file(build_prompt(text))
    if prompt_path is None:
        return None

    command = [_binary(), "-m", _model_path(), "-f", prompt_path,
               "-n", str(MAX_NEW_TOKENS), "--temp", "0"] + get_extra_args()
    try:
        returncode, stdout, stderr = _run_model(command, timeout)
    except subprocess.TimeoutExpired:
        LOGGER.warning("SLM timed out after %.1f s; using fallback", timeout)
        return None
    except OSError as error:
        LOGGER.warning("SLM could not start (%s); using fallback", error)
        return None
    finally:
        _remove_quietly(prompt_path)

    if returncode != 0:
        LOGGER.warning("SLM exited with code %s: %s", returncode, stderr.strip()[-200:])
        return None
    return extract_json(stdout)
