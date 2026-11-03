# saam/core: intent interpretation and controller

| File | Purpose |
|---|---|
| `intent_parser.py` | `parse_intent(text)` -> `{"intent": ..., "target": ...}`. Tries the SLM, falls back to rules. |
| `slm_engine.py` | Runs the small language model through llama.cpp (subprocess). Returns `None` if unavailable. |
| `response_builder.py` | `build_response(intent, vision_result)` -> sentence to speak. |
| `controller.py` | `handle_command(text)` -> response string. Uses mock vision until the real module is ready. |
| `object_labels.py` + `object_labels.txt` | The object names vision supports. Empty until the vision owner provides the list. |

Supported intents: `FIND_OBJECT`, `DESCRIBE_SCENE`, `STOP` (plus `UNKNOWN` for anything not understood).

## Run the demo (no hardware or model needed)
From the repository root:

    python -m saam.core.controller              # runs sample commands
    python -m saam.core.controller --interactive
    python -m saam.core.controller -v           # also shows which parser was used

## Run the tests
From the repository root:

    python -m unittest discover -s tests -t . -v

Test files: `test_intent_parser.py`, `test_slm_engine.py` (uses a fake llama.cpp command, including a real timeout), `test_controller.py`. Each test sets its own environment and restores it afterwards.

## Prototype default
For the Nov 5 prototype the system uses rule-based parsing and mock vision. The SLM is optional and is only used if the settings below are present.

## Using the SLM (optional)
Without these settings the rule-based fallback is used.

    export SAAM_LLAMA_BIN=llama-cli            # llama.cpp command or path
    export SAAM_MODEL_PATH=/path/to/model.gguf # e.g. Qwen2.5-0.5B-Instruct Q4_K_M
    export SAAM_SLM_TIMEOUT=10                 # seconds (a number above 0, max 60); bad values use 10
    export SAAM_LLAMA_EXTRA_ARGS="--no-display-prompt"

llama.cpp flag names vary by version. Check them with `llama-cli --help` on the board.
`SAAM_DISABLE_SLM=1` forces the rule-based parser.

## Using the real vision module later
Pass its functions in:

    handle_command(text, detect_object=real_detect_object, describe_scene=real_describe_scene)

Vision failures must be reported, not hidden:
- `detect_object()` returns a dict with an `"error"` key on failure.
- `describe_scene()` returns a list (empty means the camera worked and saw nothing) or `{"error": "CAMERA_UNAVAILABLE"}` on failure.
- Raising an exception also works; the controller says the camera is unavailable.

## STOP and the audio loop
When the user says stop, `handle_command()` calls the `on_stop` function it was given:

    handle_command(text, on_stop=stop_speaking)

`stop_speaking` is provided by the audio owner and must cut off speech that is playing. See the integration guide.

## Object labels
Paste the vision module's labels into `object_labels.txt`, one per line. Once the list has labels, asking for anything else gets "Sorry, I can't look for a ... yet." and vision is not called. Also update `ALIASES` in `intent_parser.py` so spoken names map onto those labels.

## Notes
- Python 3.8+ compatible, standard library only.
- The object vocabulary (`ALIASES`, `MULTIWORD_LABELS` in `intent_parser.py`) is an example. Match it to the vision module's labels.
- Confidence threshold (0.5) and spoken phrases live at the top of `response_builder.py`.
