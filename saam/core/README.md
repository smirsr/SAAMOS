# saam/core: intent interpretation, controller and operating modes

| File | Purpose |
|---|---|
| `intent_parser.py` | `parse_intent(text)` -> `{"intent": ..., "target": ...}`. Tries the SLM, falls back to rules. |
| `slm_engine.py` | Runs the small language model through llama.cpp (subprocess). Returns `None` if unavailable. |
| `response_builder.py` | All spoken wording: `build_response(intent, vision_result)` and the mode messages. |
| `controller.py` | The original one-shot path: `handle_command(text)` -> response string. Uses mock vision until the real module is ready. |
| `detections.py` | **New.** Accepts the agreed detection format (and the old one) and tells "not found" apart from "failed". |
| `modes.py` | **New.** `ModeController`: operating modes, Find/Scan, push-to-talk, repeat. Built on `controller.py`. |
| `modes_demo.py` | **New.** Keyboard demo of the modes (mock vision). The original demo is untouched. |
| `object_labels.py` + `object_labels.txt` | The object names vision supports. Empty until the vision owner provides the list. |

## Run the demos (no hardware or model needed)
From the repository root:

    python -m saam.core.controller              # original demo: sample commands
    python -m saam.core.controller --interactive
    python -m saam.core.modes_demo              # new: modes, buttons, Find/Scan

## Run the tests
From the repository root:

    python -m unittest discover -s tests -t . -v

Test files: `test_intent_parser.py`, `test_slm_engine.py` (fake llama.cpp command, including a real timeout), `test_controller.py`, `test_modes.py` (modes, buttons, stop safety, threads). Each test sets its own environment and restores it afterwards. No board, camera, microphone or GPIO is needed.

## Operating modes

| Mode | Meaning |
|---|---|
| `IDLE` | ready for input |
| `LISTENING` | recording a voice command (short press) |
| `FINDING` | continuously looking for one object |
| `SCANNING` | watching the camera for objects that appear or disappear |
| `READING` | one OCR request (placeholder until OCR is installed) |
| `SPEAKING` | delivering a response |

FINDING and SCANNING are *continuous*. LISTENING, READING and SPEAKING are short; when they end the controller returns to the continuous mode that was running, or to IDLE.

## Commands

| Say | Intent | What happens |
|---|---|---|
| "Where is my bottle?" | `FIND_OBJECT` | answered once (original behavior) |
| "Find my bottle" | `START_FIND` | Find Mode: announces when found, moved or lost |
| "Stop searching" | `STOP_FIND` | ends Find Mode |
| "Scan my surroundings" | `START_SCAN` | Scan Mode: describes the scene, then only meaningful changes |
| "Stop scanning" | `STOP_SCAN` | ends Scan Mode |
| "How many bottles do you see?" | `COUNT_OBJECTS` | count from vision |
| "Read this text" | `READ_TEXT` | READING mode, then speaks the text |
| "Repeat that" | `REPEAT` | replays the last spoken response |
| "Stop" | `STOP` | ends whatever is happening, immediately |
| "What objects are in front of me?" | `DESCRIBE_SCENE` | answered once (original behavior) |

Any object label works ("Find my cup", "How many people ..."), not just "bottle". Plurals are handled for counting. If `object_labels.txt` lists labels, anything else gets "Sorry, I can't look for a ... yet." and vision is not called.

## Using ModeController (for the hardware integration)

    from saam.core.modes import ModeController, ButtonEvent

    controller = ModeController(
        vision=vision,                    # detect_object(), describe_scene(), optional count_objects()
        speak=speak,                      # speak(text)
        stop_speaking=stop_speaking,      # cut off speech that is playing
        start_listening=start_listening,  # begin recording a command
        stop_listening=stop_listening,    # cancel a recording
        read_text=read_text,              # OCR: returns {"text": ...} or {"error": ...}
    )

| Call | When |
|---|---|
| `controller.on_button_event(ButtonEvent.SHORT_PRESS)` | short press: start listening |
| `controller.on_button_event(ButtonEvent.LONG_PRESS)` | long press: stop the current continuous mode |
| `controller.on_button_event(ButtonEvent.REPEAT)` | repeat event (double press later): replay last response |
| `controller.on_key("s" / "l" / "r" / " ")` | keyboard stand-in for the button |
| `controller.handle_text(text)` | speech recognition finished: runs the command and speaks the answer |
| `controller.tick()` | call about twice a second so Find/Scan Mode keeps working |
| `controller.stop()` | stop everything now (safe from any thread) |
| `controller.speech_finished()` | only if `speak()` does not block: call when audio ends |
| `controller.mode`, `.activity`, `.last_response` | read the current state |

GPIO is never touched here. The hardware owner turns presses into `ButtonEvent` values.

A simple host loop:

    while True:
        controller.tick()
        time.sleep(0.5)
    # elsewhere: the button thread calls controller.on_button_event(...)
    # and the audio thread calls controller.handle_text(...) when a command was recognized

## Why it cannot get stuck
- There are no loops inside `modes.py`. Each `tick()` does one small step, so the host stays in control.
- Find Mode stops itself after 120 s, Scan Mode after 300 s, listening after 10 s, and speech state after at most 30 s (`ModeSettings`).
- Three camera errors in a row end Find/Scan Mode with a spoken message.
- `stop()` never raises, never waits for speech, the camera or OCR, and any late result from them is thrown away.
- Short flickers are ignored: a change must be seen on 2 ticks in a row (`confirm_ticks`).

## Vision format (agreed with the vision owner)
A detection:

    {"label": "bottle", "confidence": 0.85, "position": "left", "bbox": [x1, y1, x2, y2]}

`position` is `"left"`, `"center"` or `"right"`. `bbox` is kept but the controller does not use it. The earlier format (`{"object": ..., "detected": True, ...}`) is still accepted. `detections.py` converts both into one shape, so the rest of the code never cares.

| Function | Returns |
|---|---|
| `detect_object(target)` | one detection dict (or a list; the best match is used). `None`, `{}` or `[]` when the object is not in view. `{"error": "..."}` if the camera or model failed. |
| `describe_scene()` | a list of detections (`[]` = the camera worked and saw nothing), or `{"error": "..."}` on failure. Use one list entry per object instance. |
| `count_objects(target)` | optional: `{"label": "bottle", "count": 2}` or `{"error": "..."}`. If it does not exist, counting uses `describe_scene()`. |

"Not in view" and "failed" are different on purpose: a failure is spoken as a camera problem and is never reported as "your bottle is not here". Raising an exception also counts as a failure. A detection with a different label than the one asked for is treated as not found. Find Mode polls `detect_object()` and Scan Mode polls `describe_scene()` about twice a second; "appeared / disappeared / moved" is worked out by comparing consecutive results.

## OCR interface (placeholder)
`read_text() -> {"text": "..."}` or `{"error": "..."}`. The default placeholder returns `{"error": "OCR_NOT_INSTALLED"}`, so "Read this text" says "Reading text isn't set up yet."

## Using the SLM (optional)
For the prototype the system uses rule-based parsing and mock vision. The SLM is only used if these are set:

    export SAAM_LLAMA_BIN=llama-cli            # llama.cpp command or path
    export SAAM_MODEL_PATH=/path/to/model.gguf # e.g. Qwen2.5-0.5B-Instruct Q4_K_M
    export SAAM_SLM_TIMEOUT=10                 # seconds (a number above 0, max 60); bad values use 10
    export SAAM_LLAMA_EXTRA_ARGS="--no-display-prompt"

llama.cpp flag names vary by version. Check them with `llama-cli --help` on the board. `SAAM_DISABLE_SLM=1` forces the rule-based parser.

## Known limitations
- Only one continuous mode runs at a time. Starting Scan replaces Find, and the reverse.
- "Find ..." now starts Find Mode. Questions like "Where is ...?" are still answered once. In the old `handle_command()` path (no mode controller) "Find ..." is answered once.
- Change announcements depend on vision being steady. Very noisy detections will be slow to announce (a change must last 2 ticks).
- Counting from `describe_scene()` assumes one list entry per object instance.
- `bbox` is passed through but not used yet (for example, it could later give size or distance).
- Object names must match between what the user says and the vision labels (`ALIASES` in `intent_parser.py`, `object_labels.txt`).
- If `speak()` blocks, a stop request still works only if the button runs on a different thread than the speech.
- READING waits for the OCR function to return; the OCR owner should keep it short or add its own timeout.
- With the SLM switched on, new commands depend on the model's output; the rules remain the default.
- Object names such as "water bottle" -> "bottle" come from `ALIASES` in `intent_parser.py`, which still needs to be matched to the vision labels.
