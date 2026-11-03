# Integration Guide

**Status:** Working draft.  Will be updated if implementation requires changes.

## Overview
An offline AI accessibility assistant for blind and visually impaired users. Pipeline:

User speaks -> Speech recognition -> SLM/controller -> Computer vision -> Response generation -> Text-to-speech

**Goal:** a preliminary prototype for Round 2 (November 5), focused on three core features: object recognition and localization, offline voice interaction, and spoken environmental feedback. Reliability comes before any additional features.

Each module is a Python function with a defined input and output so everyone can build and test independently. All modules run as plain function calls in one Python process.

## Ownership
| Component | Owner |
|---|---|
| Speech recognition + audio feedback | Smridhi |
| SLM + controller + response generation | Arianna |
| Computer vision | Marcos |
| Hardware + deployment (Linux setup) | Sarah |

## Conventions
- All modules are written in Python 3, compatible with Python 3.8 or newer until the TI Linux image version is confirmed. Bash is used only for Linux setup and deployment scripts, when needed. Type hints use `typing` (`List`, `Dict`).
- All final features must operate offline. No cloud APIs for models or speech.
- Files and functions use `snake_case`; classes use `PascalCase`.
- Modules exchange Python dicts or JSON-compatible objects only.
- Camera index and audio device are never hardcoded. They come from one shared configuration file, with default devices as a fallback. The hardware/deployment owner maintains it.
- Dependencies are recorded with versions; use a virtual environment where supported.
- Every module ships with a standalone test (e.g. `tests/test_intent_parser.py`) so it can be checked without the other modules.
- Object names sent to vision must be labels the vision module supports. The list lives in `saam/core/object_labels.txt`, maintained with the vision owner.
- The prototype starts with rule-based parsing and mock vision. The SLM is optional.


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
| `UNKNOWN` | `null` | anything not understood |

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

- `position` values: `"left"`, `"center"`, `"right"`
- Not found: `detected: false`, `position: null`, `confidence: 0.0`
- Camera or model error: `{"object": "bottle", "detected": false, "error": "CAMERA_UNAVAILABLE"}`
- The controller treats any detection with confidence below 0.5 as "not found".
- Model details (ONNX, TFLite, OpenCV) are hidden inside this module. Callers only see the dict.

**Function:** `describe_scene() -> List[Dict]`
- Input: none
- Output: list of detections, each in the same format as above. An empty list means the camera worked and nothing was detected.
- On camera or model error: return `{"error": "CAMERA_UNAVAILABLE"}` (raising an exception also works). Never return an empty list for a failure.

## 4. Response generation
**Owner:** Arianna

**Function:** `build_response(intent: Dict, vision_result) -> str`
- `FIND_OBJECT`, found: `"Your bottle is on the left."`
- `FIND_OBJECT`, not found: `"I couldn't find your bottle."`
- `DESCRIBE_SCENE`: summarizes the detected objects and their positions
- `STOP`: no vision call; speech stops, any pending request is cancelled, and a short confirmation (or an empty string) is returned
- `UNKNOWN` or error: `"Sorry, I didn't catch that. Could you repeat it?"`

Rules, since users hear responses rather than read them:
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

## STOP and the audio loop (proposed)
- The audio module (leader) provides `stop_speaking() -> None`, which immediately cuts off speech that is playing.
- The controller calls it when it hears STOP: `handle_command(text, on_stop=stop_speaking)`. The controller never touches the speaker or microphone.
- For STOP to work while the system is talking, the audio module must keep listening during playback (for example, play speech in the background). How is up to the audio owner.
- Main loop: `recognize_speech()` -> `handle_command(...)` -> `speak(reply)`, then back to listening.
- If `stop_speaking()` fails, the controller still returns its confirmation and logs the error.

## Error handling rules
- Modules never crash the pipeline. They return the failure values above.
- Every failure ends in a message to the user.
- If the SLM is unavailable or too slow, the rule-based fallback keeps the rest of the system working.

## Assumptions
These are working decisions. They will be adjusted if implementation shows a problem.
- All modules run as plain Python function calls in one process.
- `UNKNOWN` is a valid fourth intent, used when a command isn't understood.
- `STOP` stops any speech currently playing and cancels the pending request. The controller returns a short confirmation or an empty string.
- `describe_scene() -> List[Dict]` is the vision function used for `DESCRIBE_SCENE`. Each item uses the same format as `detect_object()`.
- A detection with confidence below 0.5 is treated as "not found".
- Code is written to run on Python 3.8 or newer until the TI Linux image version is confirmed.
- Camera index and audio device come from one shared configuration file, with default devices as a fallback. The hardware/deployment owner maintains it.