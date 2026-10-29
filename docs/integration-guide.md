# Integration Guide

**Status:** DRAFT, proposed for team approval

## Overview
An offline AI accessibility assistant for blind and visually impaired users. Pipeline:

User speaks -> Speech recognition -> SLM/controller -> Computer vision -> Response generation -> Text-to-speech

**Goal:** a preliminary prototype for Round 2 (November 5), focused on three core features: object recognition and localization, offline voice interaction, and spoken environmental feedback. Reliability comes before any additional features.

Each module is a Python function with a defined input and output so everyone can build and test independently. This draft assumes all modules run as plain function calls in one Python process (see Open questions).

## Ownership
| Component | Owner |
|---|---|
| Speech recognition + audio feedback | Smridhi |
| SLM + controller + response generation | Arianna |
| Computer vision | Marcos |
| Hardware + deployment (Linux setup) | Sarah |

## Conventions
- All modules are written in Python 3 (exact minor version to be finalized after checking the TI Linux image). Bash is used only for Linux setup and deployment scripts, when needed. Type hints use `typing` (`List`, `Dict`) so they work on older Python 3 versions.
- All final features must operate offline. No cloud APIs for models or speech.
- Files and functions use `snake_case`; classes use `PascalCase`.
- Modules exchange Python dicts or JSON-compatible objects only.
- Camera index and audio device are never hardcoded. They come from configuration, with a fallback.
- Dependencies are recorded with versions; use a virtual environment where supported.
- Every module ships with a standalone test (e.g. `tests/test_intent_parser.py`) so it can be checked without the other modules.


## 1. Audio module (speech recognition)
**Owner:** Smridhi
**Function:** `recognize_speech() -> str`
- Input: microphone audio
- Output: recognized text as a plain string, e.g. `"Where is my water bottle?"`
- The speech engine's own result format (for example Vosk's JSON) is unwrapped inside this module. Callers only ever see the plain string.
- On failure (no speech, unintelligible, mic error): return `""`. The controller treats this as no input and asks the user to repeat.

## 2. SLM/controller module
**Owner:** Arianna

**Function:** `parse_intent(text: str) -> Dict`
- Input: recognized text
- Output: structured intent

```json
{
  "intent": "FIND_OBJECT",
  "target": "bottle"
}
```

Supported intents:

| Intent | Target | Example command |
|---|---|---|
| `FIND_OBJECT` | object name | "Where is my water bottle?" |
| `DESCRIBE_SCENE` | `null` | "What objects are in front of me?" |
| `STOP` | `null` | "Stop." |
| `UNKNOWN` (proposed) | `null` | anything not understood |

- The SLM is tried first. If it fails, times out, or returns invalid output, a keyword/rule-based fallback parser produces the same output format. Callers never need to know which method was used.
- `READ_TEXT` may be added later if OCR is implemented.

**Controller entry point:** `handle_command(text: str) -> str`
- Input: recognized text from the audio module
- Output: response string to send to text-to-speech
- Flow: parse intent, call vision if needed, build the response.
- During development, the controller can use mock vision results so it can be tested on its own.

## 3. Vision module
**Owner:** Marcos

**Function:** `detect_object(target: str) -> Dict`
- Input: target object name, e.g. `"bottle"`
- Output:

```json
{
  "object": "bottle",
  "detected": true,
  "position": "left",
  "confidence": 0.91
}
```

- `position` values (proposed): `"left"`, `"center"`, `"right"`
- Not found: `detected: false`, `position: null`, `confidence: 0.0`
- Camera or model error: `{"object": "bottle", "detected": false, "error": "CAMERA_UNAVAILABLE"}`
- Model details (ONNX, TFLite, OpenCV) are hidden inside this module. Callers only see the dict.

**Function (proposed):** `describe_scene() -> List[Dict]`
- Input: none
- Output: list of detections, each in the same format as above
- Empty list if nothing is detected or on error

## 4. Response generation
**Owner:** Arianna

**Function:** `build_response(intent: Dict, vision_result) -> str`
- `FIND_OBJECT`, found: `"Your bottle is on the left."`
- `FIND_OBJECT`, not found: `"I couldn't find your bottle."`
- `DESCRIBE_SCENE`: summarizes the detected objects and their positions
- `STOP`: no vision call; returns a short confirmation or an empty string
- `UNKNOWN` or error: `"Sorry, I didn't catch that. Could you repeat it?"`

Proposed rules, since users hear responses rather than read them:
- Keep responses short and unambiguous.
- `DESCRIBE_SCENE` mentions only a few objects, closest or most confident first.
- `STOP` is handled immediately so the user can interrupt speech.

## 5. Audio feedback module (text-to-speech)
**Owner:** Smridhi
**Function:** `speak(text: str) -> None`
- Input: response string
- Output: spoken audio
- The speech engine (for example eSpeak NG or Piper) is hidden inside this module so it can be swapped without affecting other code.
- On failure: log the error and print the text to the console.

## Error handling rules
- Modules never crash the pipeline. They return the failure values above.
- Every failure ends in a message to the user.
- If the SLM is unavailable or too slow, the rule-based fallback keeps the rest of the system working.

## Open questions
- Are all modules plain Python function calls in one process?
- Is `UNKNOWN` acceptable as a fourth intent?
- What exactly should `STOP` do (stop speech, stop the current task)?
- Is `describe_scene()` the right vision interface, and what is its output format?
- Should confidence below a threshold (e.g. 0.5) count as "not found"?
- Which Python version does the TI Linux image support?
- For Sarah: how are the camera index and audio device configured, and how do other modules read that configuration?