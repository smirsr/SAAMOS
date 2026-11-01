# saam/core: intent interpretation and controller

| File | Purpose |
|---|---|
| `intent_parser.py` | `parse_intent(text)` -> `{"intent": ..., "target": ...}`. Tries the SLM, falls back to rules. |
| `slm_engine.py` | Runs the small language model through llama.cpp (subprocess). Returns `None` if unavailable. |
| `response_builder.py` | `build_response(intent, vision_result)` -> sentence to speak. |
| `controller.py` | `handle_command(text)` -> response string. Uses mock vision until the real module is ready. |

Supported intents: `FIND_OBJECT`, `DESCRIBE_SCENE`, `STOP` (plus `UNKNOWN` for anything not understood).

## Run the demo (no hardware or model needed)
From the repository root:

    python -m saam.core.controller              # runs sample commands
    python -m saam.core.controller --interactive
    python -m saam.core.controller -v           # also shows which parser was used

## Run the tests
From the repository root:

    python -m unittest discover -s tests -t . -v

## Using the SLM (optional)
Without these settings the rule-based fallback is used.

    export SAAM_LLAMA_BIN=llama-cli            # llama.cpp command or path
    export SAAM_MODEL_PATH=/path/to/model.gguf # e.g. Qwen2.5-0.5B-Instruct Q4_K_M
    export SAAM_SLM_TIMEOUT=10                 # seconds before falling back
    export SAAM_LLAMA_EXTRA_ARGS="--no-display-prompt"

llama.cpp flag names vary by version. Check them with `llama-cli --help` on the board.
`SAAM_DISABLE_SLM=1` forces the rule-based parser.

## Using the real vision module later
Pass its functions in:

    handle_command(text, detect_object=real_detect_object, describe_scene=real_describe_scene)

## Notes
- Python 3.8+ compatible, standard library only.
- The object vocabulary (`ALIASES`, `MULTIWORD_LABELS` in `intent_parser.py`) is an example. Match it to the vision module's labels.
- Confidence threshold (0.5) and spoken phrases live at the top of `response_builder.py`.
