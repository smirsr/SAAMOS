"""Small language model (SLM) engine, run through llama.cpp.

This module hides how the model is run. The rest of the system only calls
slm_parse(), which returns a dict like {"intent": "STOP", "target": None} or
None when the model is unavailable, too slow, or gives unusable output. The
caller (intent_parser.py) then falls back to the rule-based parser.

Nothing here is hardcoded to one machine. The llama.cpp binary and the model
file are set with environment variables:

    SAAM_LLAMA_BIN         llama.cpp command or path, e.g. llama-cli
    SAAM_MODEL_PATH        path to the .gguf model file
    SAAM_SLM_TIMEOUT       seconds to wait before giving up (default 10)
    SAAM_LLAMA_EXTRA_ARGS  extra llama.cpp flags (default: --no-display-prompt)
    SAAM_DISABLE_SLM       set to 1 to skip the model entirely

llama.cpp flag names change between versions. Check them with
`llama-cli --help` on the board and adjust SAAM_LLAMA_EXTRA_ARGS if needed.

Everything runs locally. No network or cloud API is used.
"""

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)

ENV_BIN = "SAAM_LLAMA_BIN"
ENV_MODEL = "SAAM_MODEL_PATH"
ENV_TIMEOUT = "SAAM_SLM_TIMEOUT"
ENV_EXTRA_ARGS = "SAAM_LLAMA_EXTRA_ARGS"
ENV_DISABLE = "SAAM_DISABLE_SLM"

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_EXTRA_ARGS = ["--no-display-prompt"]
MAX_NEW_TOKENS = 48

PROMPT_TEMPLATE = """You turn a spoken command into JSON.
Intents: FIND_OBJECT (needs a target object name), DESCRIBE_SCENE, STOP, UNKNOWN.
Reply with the JSON only.

Command: Where is my water bottle?
JSON: {"intent": "FIND_OBJECT", "target": "bottle"}
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


def _timeout() -> float:
    try:
        return float(os.environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT_S))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _extra_args() -> List[str]:
    raw = os.environ.get(ENV_EXTRA_ARGS)
    if raw is None:
        return list(DEFAULT_EXTRA_ARGS)
    return shlex.split(raw)


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


def slm_parse(text: str) -> Optional[Dict]:
    """Ask the model for an intent. Returns a dict, or None on any problem."""
    if not is_available():
        return None

    command = [_binary(), "-m", _model_path(), "-p", build_prompt(text),
               "-n", str(MAX_NEW_TOKENS), "--temp", "0"] + _extra_args()
    try:
        # stdin is closed so llama.cpp can never wait for interactive input.
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_timeout(),
        )
    except subprocess.TimeoutExpired:
        LOGGER.warning("SLM timed out after %.1f s; using fallback", _timeout())
        return None
    except OSError as error:
        LOGGER.warning("SLM could not start (%s); using fallback", error)
        return None

    if completed.returncode != 0:
        LOGGER.warning("SLM exited with code %s: %s", completed.returncode,
                       (completed.stderr or "").strip()[-200:])
        return None
    return extract_json(completed.stdout)
