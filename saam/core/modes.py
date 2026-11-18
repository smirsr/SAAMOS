"""Operating modes and push-to-talk for SAAMOS.

This builds on the existing controller (controller.py). One-shot questions such
as "Where is my bottle?" are answered by the same code as before; this module
adds the things that keep running or react to buttons:

    IDLE       ready for input
    LISTENING  recording a voice command (started by a short press)
    FINDING    continuously looking for one object ("Find my bottle")
    SCANNING   watching the camera for objects that appear or disappear
    READING    sending a camera frame to OCR (placeholder until OCR is installed)
    SPEAKING   delivering an audio response

How it is meant to be used (the host program is a simple loop):

    controller = ModeController(vision=..., speak=..., stop_speaking=...,
                                start_listening=..., read_text=...)
    controller.on_button_event(ButtonEvent.SHORT_PRESS)   # from the GPIO code
    controller.handle_text("Find my bottle")              # from speech recognition
    controller.tick()                                     # call about twice a second

Design rules (so it is easy to explain and cannot get stuck)
------------------------------------------------------------
* No loops inside this file. Find and Scan do ONE small step each time tick()
  is called, so the host stays in charge and nothing can spin forever.
* Every continuous mode has a time limit, every wait has a timeout, and
  repeated camera errors end the mode. Whatever happens, the controller
  returns to IDLE on its own.
* stop() always works: it never raises, it can be called from any thread (for
  example the button thread) at any time, and it never waits for speech or the
  camera to finish.
* GPIO is never touched here. The hardware owner turns button presses into
  ButtonEvent values and passes them to on_button_event().
"""

import enum
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

from . import response_builder as rb
from .controller import MockVision, respond_to_intent
from .detections import label_of, normalize_find_result, normalize_scene_result
from .intent_parser import (COUNT_OBJECTS, READ_TEXT, REPEAT, START_FIND,
                            START_SCAN, STOP, STOP_FIND, STOP_SCAN, parse_intent)
from .object_labels import is_supported_label

LOGGER = logging.getLogger(__name__)


class Mode(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    FINDING = "FINDING"
    SCANNING = "SCANNING"
    READING = "READING"
    SPEAKING = "SPEAKING"


class ButtonEvent(enum.Enum):
    """What the physical button (or a keyboard key) did."""
    SHORT_PRESS = "short_press"   # start listening for a voice command
    LONG_PRESS = "long_press"     # stop the current continuous mode
    REPEAT = "repeat"             # replay the previous spoken response


_BUTTON_NAMES = {
    "short": ButtonEvent.SHORT_PRESS, "short_press": ButtonEvent.SHORT_PRESS,
    "press": ButtonEvent.SHORT_PRESS,
    "long": ButtonEvent.LONG_PRESS, "long_press": ButtonEvent.LONG_PRESS,
    "hold": ButtonEvent.LONG_PRESS,
    "repeat": ButtonEvent.REPEAT, "double": ButtonEvent.REPEAT,
    "double_press": ButtonEvent.REPEAT,
}
# Keyboard stand-ins for the button while there is no hardware.
KEYBOARD_KEYS = {" ": ButtonEvent.SHORT_PRESS, "s": ButtonEvent.SHORT_PRESS,
                 "l": ButtonEvent.LONG_PRESS, "r": ButtonEvent.REPEAT}


def parse_button_event(value) -> Optional[ButtonEvent]:
    """Accept a ButtonEvent, an event name ("long_press"), or a key ("l")."""
    if isinstance(value, ButtonEvent):
        return value
    if not isinstance(value, str) or value == "":
        return None
    if value.strip() == "":                     # the space bar
        return KEYBOARD_KEYS[" "]
    key = value.strip().lower()
    return _BUTTON_NAMES.get(key) or KEYBOARD_KEYS.get(key)


@dataclass
class ModeSettings:
    """Time limits and tuning. The defaults suit the prototype."""
    find_timeout_s: float = 120.0      # Find Mode stops itself after this long
    scan_timeout_s: float = 300.0      # Scan Mode stops itself after this long
    listening_timeout_s: float = 10.0  # give up if no command arrives
    speech_timeout_s: float = 30.0     # longest a response may stay SPEAKING
    confirm_ticks: int = 2             # a change must be seen this many ticks in a row
    max_vision_errors: int = 3         # consecutive camera errors before giving up
    speech_blocks: bool = True         # True: speak() returns when speech ends


def placeholder_read_text() -> Dict:
    """Stand-in for OCR until it is installed."""
    return {"error": "OCR_NOT_INSTALLED"}


def scene_counts(detections) -> Dict[str, int]:
    """{label: how many confident detections} for a describe_scene() result."""
    counts: Dict[str, int] = {}
    for detection in detections:
        if rb.is_confident_detection(detection):
            label = label_of(detection)
            counts[label] = counts.get(label, 0) + 1
    return counts


def count_from_scene(target: str, scene) -> Dict:
    """Count one label using describe_scene() when vision has no count function."""
    scene = normalize_scene_result(scene)
    if not isinstance(scene, list):
        return scene                                    # an error dict
    return {"label": target, "object": target, "count": scene_counts(scene).get(target, 0)}


def _debounced(confirmed, pending, observed, needed: int):
    """Only accept a change after it has been seen `needed` times in a row.

    This stops one flickering camera frame from being announced.
    Returns (confirmed, pending, changed).
    """
    if observed == confirmed:
        return confirmed, None, False
    count = pending[1] + 1 if pending is not None and pending[0] == observed else 1
    if count >= needed:
        return observed, None, True
    return confirmed, (observed, count), False


class ModeController:
    """Operating modes, push-to-talk and continuous Find/Scan on top of the controller.

    vision           object with detect_object(target) and describe_scene(),
                     and optionally count_objects(target). Defaults to MockVision.
    speak            speak(text) from the audio module (should return quickly
                     unless settings.speech_blocks is True and speech is short).
    stop_speaking    stop_speaking(): cut off speech that is playing.
    start_listening  start_listening(): begin recording a voice command; the
                     audio side later calls handle_text() with the result.
    stop_listening   stop_listening(): cancel a recording in progress.
    read_text        OCR function returning {"text": ...} or {"error": ...}.
    clock            time source (replaced by a fake clock in tests).
    """

    def __init__(self, vision=None, speak: Optional[Callable[[str], None]] = None,
                 stop_speaking: Optional[Callable[[], None]] = None,
                 start_listening: Optional[Callable[[], None]] = None,
                 stop_listening: Optional[Callable[[], None]] = None,
                 read_text: Optional[Callable[[], Dict]] = None,
                 settings: Optional[ModeSettings] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._vision = vision if vision is not None else MockVision()
        self._speak_cb = speak
        self._stop_speaking_cb = stop_speaking
        self._start_listening_cb = start_listening
        self._stop_listening_cb = stop_listening
        self._read_text_cb = read_text if read_text is not None else placeholder_read_text
        self._settings = settings if settings is not None else ModeSettings()
        self._clock = clock

        # Buttons may arrive on another thread, so state changes take this lock.
        # Callbacks (speak, vision, OCR) are always called OUTSIDE the lock so
        # a slow one can never block a stop request.
        self._lock = threading.RLock()
        self._mode = Mode.IDLE
        self._mode_since = clock()
        self._history: Deque[Mode] = deque([Mode.IDLE], maxlen=200)

        # Counters used to notice "this was cancelled while I was busy":
        self._epoch = 0         # changes whenever speech/listening/OCR is cut off
        self._activity_id = 0   # changes whenever Find/Scan starts or ends

        self._activity: Optional[Mode] = None   # FINDING, SCANNING or None
        self._activity_since = 0.0
        self._vision_errors = 0
        self._find_target: Optional[str] = None
        self._find_state: Optional[str] = None            # confirmed position, None = unseen
        self._find_pending: Optional[Tuple] = None
        self._find_seen = False                           # ever seen during this search
        self._scene: Optional[Dict[str, int]] = None      # confirmed {label: count}
        self._scene_pending: Optional[Tuple] = None

        self._last_response: Optional[str] = None
        self._speech_deadline = 0.0

    # ------------------------------------------------------------------
    # Read-only state (handy for tests, logging and the demo)
    # ------------------------------------------------------------------
    @property
    def mode(self) -> Mode:
        with self._lock:
            return self._mode

    @property
    def activity(self) -> Optional[Mode]:
        """The continuous mode (FINDING or SCANNING) that is running, if any."""
        with self._lock:
            return self._activity

    @property
    def last_response(self) -> Optional[str]:
        with self._lock:
            return self._last_response

    def mode_history(self) -> List[Mode]:
        with self._lock:
            return list(self._history)

    # ------------------------------------------------------------------
    # Small internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _call(callback, *args):
        """Call a host callback. A broken callback is logged, never raised."""
        if callback is None:
            return None
        try:
            return callback(*args)
        except Exception:
            LOGGER.exception("callback %r failed", callback)
            return None

    def _ask_vision(self, name: str, *args):
        """Call a vision function; any failure becomes an error dict."""
        function = getattr(self._vision, name, None)
        if not callable(function):
            return {"error": "VISION_UNAVAILABLE"}
        try:
            return function(*args)
        except Exception:
            LOGGER.exception("vision %s failed", name)
            return {"error": "VISION_ERROR"}

    def _set_mode(self, mode: Mode) -> None:
        if mode is not self._mode:
            LOGGER.debug("mode %s -> %s", self._mode.value, mode.value)
            self._mode = mode
            self._mode_since = self._clock()
            self._history.append(mode)

    def _resume_mode(self) -> Mode:
        """Where to go after a short mode (listening/speaking/reading) ends."""
        return self._activity if self._activity is not None else Mode.IDLE

    def _leave(self, expected: Mode) -> None:
        if self._mode is expected:
            self._set_mode(self._resume_mode())

    def _begin_activity(self, mode: Mode, target: Optional[str] = None) -> None:
        """Start FINDING or SCANNING (replacing any earlier one). Lock held."""
        self._activity_id += 1
        self._activity = mode
        self._activity_since = self._clock()
        self._vision_errors = 0
        self._find_target = target
        self._find_state, self._find_pending, self._find_seen = None, None, False
        self._scene, self._scene_pending = None, None
        self._set_mode(mode)

    def _clear_activity(self) -> None:
        """Forget the running activity. Lock held."""
        self._activity_id += 1       # any poll still in progress is now ignored
        self._activity = None
        self._find_target = None
        self._find_state, self._find_pending, self._find_seen = None, None, False
        self._scene, self._scene_pending = None, None

    def _end_activity(self, only: Optional[Mode] = None) -> bool:
        """End Find/Scan. Returns False if that activity was not running."""
        with self._lock:
            if self._activity is None or (only is not None and self._activity is not only):
                return False
            self._clear_activity()
            if self._mode in (Mode.FINDING, Mode.SCANNING):
                self._set_mode(Mode.IDLE)
            return True

    # ------------------------------------------------------------------
    # Speaking
    # ------------------------------------------------------------------
    def _say(self, text: Optional[str], remember: bool = True) -> None:
        """Speak a response. It becomes the "last response" unless remember=False."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            if remember:
                self._last_response = text
            interrupted = self._mode is Mode.SPEAKING
            self._epoch += 1
            epoch = self._epoch
            self._set_mode(Mode.SPEAKING)
            limit = min(self._settings.speech_timeout_s, 2.0 + 0.08 * len(text))
            self._speech_deadline = self._clock() + limit
        if interrupted:
            self._call(self._stop_speaking_cb)       # never talk over ourselves
        self._call(self._speak_cb, text)
        if self._settings.speech_blocks:
            with self._lock:
                # If stop() ran while we were speaking, the epoch changed and
                # we must not touch the mode.
                if self._epoch == epoch:
                    self._leave(Mode.SPEAKING)

    def speech_finished(self) -> None:
        """Tell the controller that spoken audio ended (when speech does not block)."""
        with self._lock:
            self._leave(Mode.SPEAKING)

    def repeat_last(self) -> str:
        """Replay the most recent spoken response (it stays the "last response")."""
        with self._lock:
            text = self._last_response
            was_listening = self._mode is Mode.LISTENING
        if was_listening:
            self._call(self._stop_listening_cb)
        if text is None:
            self._say(rb.NOTHING_TO_REPEAT_MESSAGE, remember=False)
            return rb.NOTHING_TO_REPEAT_MESSAGE
        self._say(text, remember=False)
        return text

    # ------------------------------------------------------------------
    # Stopping (must always work)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """End whatever is happening and return to IDLE. Safe to call anytime.

        Cuts off speech, cancels listening, ends Find/Scan and abandons any
        camera or OCR request still in progress. Never raises and never waits.
        """
        with self._lock:
            was_listening = self._mode is Mode.LISTENING
            self._epoch += 1
            self._clear_activity()
            self._set_mode(Mode.IDLE)
        self._call(self._stop_speaking_cb)
        if was_listening:
            self._call(self._stop_listening_cb)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    def start_listening(self) -> None:
        """Short press: stop talking and start recording a voice command."""
        with self._lock:
            if self._mode in (Mode.LISTENING, Mode.READING):
                return
            interrupted = self._mode is Mode.SPEAKING
            self._epoch += 1
            self._set_mode(Mode.LISTENING)
        if interrupted:
            self._call(self._stop_speaking_cb)
        if self._start_listening_cb is not None:
            try:
                self._start_listening_cb()
            except Exception:
                LOGGER.exception("start_listening callback failed")
                with self._lock:
                    self._leave(Mode.LISTENING)
                self._say(rb.LISTEN_FAILED_MESSAGE, remember=False)

    def on_button_event(self, event) -> Optional[str]:
        """Handle a button press. Accepts a ButtonEvent, its name, or a key.

            SHORT_PRESS -> start listening for a voice command
            LONG_PRESS  -> stop the current continuous mode (and any speech)
            REPEAT      -> replay the previous spoken response

        Returns the text that was spoken, if any. Unknown events are ignored.
        """
        parsed = parse_button_event(event)
        if parsed is None:
            LOGGER.warning("ignoring unknown button event %r", event)
            return None
        if parsed is ButtonEvent.SHORT_PRESS:
            self.start_listening()
            return None
        if parsed is ButtonEvent.LONG_PRESS:
            with self._lock:
                activity = self._activity
            self.stop()
            message = {Mode.FINDING: rb.FIND_STOPPED_MESSAGE,
                       Mode.SCANNING: rb.SCAN_STOPPED_MESSAGE}.get(activity)
            if message:
                self._say(message)
            return message
        return self.repeat_last()

    def on_key(self, key: str) -> Optional[str]:
        """Keyboard stand-in for the button: space or s, l, r."""
        return self.on_button_event(key)

    # ------------------------------------------------------------------
    # Voice commands
    # ------------------------------------------------------------------
    def handle_text(self, text: str) -> str:
        """Handle one recognized voice command and speak the answer.

        Normally the mode goes LISTENING -> SPEAKING -> back to IDLE (or to the
        Find/Scan mode that was already running).
        """
        with self._lock:
            if self._mode is Mode.LISTENING:
                self._mode_since = self._clock()   # a command arrived: restart the timer
        intent = parse_intent(text)
        if intent["intent"] == REPEAT:
            reply = self.repeat_last()
        else:
            reply = self._dispatch(intent)
            self._say(reply)
        with self._lock:
            if self._mode is Mode.LISTENING:       # nothing was spoken: do not stay in LISTENING
                self._leave(Mode.LISTENING)
        return reply

    def _dispatch(self, intent: Dict) -> str:
        name, target = intent["intent"], intent.get("target")
        if name == STOP:
            self.stop()
            return rb.STOP_MESSAGE
        if name == STOP_FIND:
            return rb.FIND_STOPPED_MESSAGE if self._end_activity(Mode.FINDING) \
                else rb.NOT_SEARCHING_MESSAGE
        if name == STOP_SCAN:
            return rb.SCAN_STOPPED_MESSAGE if self._end_activity(Mode.SCANNING) \
                else rb.NOT_SCANNING_MESSAGE
        if name == START_FIND:
            return self._start_find(target)
        if name == START_SCAN:
            with self._lock:
                self._begin_activity(Mode.SCANNING)
            return rb.SCAN_STARTED_MESSAGE
        if name == COUNT_OBJECTS:
            return self._count(target)
        if name == READ_TEXT:
            return self._read()
        # FIND_OBJECT, DESCRIBE_SCENE and UNKNOWN use the original one-shot code.
        return respond_to_intent(intent,
                                 lambda t: self._ask_vision("detect_object", t),
                                 lambda: self._ask_vision("describe_scene"))

    def _start_find(self, target: Optional[str]) -> str:
        if not target:
            return rb.UNKNOWN_MESSAGE
        if not is_supported_label(target):         # not a label vision knows
            return rb.unsupported_object_response(target)
        with self._lock:
            self._begin_activity(Mode.FINDING, target)
        return rb.find_started_message(target)

    def _count(self, target: Optional[str]) -> str:
        if not target:
            return rb.UNKNOWN_MESSAGE
        if not is_supported_label(target):
            return rb.unsupported_object_response(target)
        if callable(getattr(self._vision, "count_objects", None)):
            result = self._ask_vision("count_objects", target)
        else:                                       # fall back to counting the scene
            result = count_from_scene(target, self._ask_vision("describe_scene"))
        return rb.build_response({"intent": COUNT_OBJECTS, "target": target}, result)

    def _read(self) -> str:
        """READING mode: one OCR request. If stop() runs meanwhile, stay quiet."""
        with self._lock:
            self._epoch += 1
            epoch = self._epoch
            self._set_mode(Mode.READING)
        try:
            result = self._read_text_cb()
        except Exception:
            LOGGER.exception("OCR failed")
            result = {"error": "OCR_ERROR"}
        with self._lock:
            if self._epoch != epoch:
                return ""
            self._leave(Mode.READING)
        return rb.read_text_response(result)

    # ------------------------------------------------------------------
    # Continuous modes: one step per tick()
    # ------------------------------------------------------------------
    def tick(self) -> Optional[str]:
        """Do one small step. Call this regularly (about twice a second).

        Enforces every time limit, then (if Find or Scan is running) asks
        vision once and announces anything meaningful that changed. Returns
        the announcement, or None.
        """
        message: Optional[str] = None
        plan = None
        with self._lock:
            now = self._clock()
            # --- safety nets: no mode may wait forever ---
            if self._mode is Mode.LISTENING:
                if now - self._mode_since <= self._settings.listening_timeout_s:
                    return None
                self._leave(Mode.LISTENING)
                message = rb.LISTEN_TIMEOUT_MESSAGE
            elif self._mode is Mode.SPEAKING:
                if now < self._speech_deadline:
                    return None
                self._leave(Mode.SPEAKING)
            elif self._mode is Mode.READING:
                return None

            if message is None and self._activity is not None:
                limit = (self._settings.find_timeout_s if self._activity is Mode.FINDING
                         else self._settings.scan_timeout_s)
                if now - self._activity_since > limit:
                    if self._activity is Mode.FINDING:
                        message = rb.find_timeout_message(self._find_target, self._find_seen)
                    else:
                        message = rb.SCAN_TIMEOUT_MESSAGE
                    self._end_activity()
                else:
                    plan = (self._activity, self._activity_id, self._find_target)

        if message is not None:
            self._say(message)
            return message
        if plan is None:
            return None

        activity, activity_id, target = plan
        # Ask vision OUTSIDE the lock so a slow camera cannot block stop().
        if activity is Mode.FINDING:
            result = self._ask_vision("detect_object", target)
        else:
            result = self._ask_vision("describe_scene")

        with self._lock:
            if self._activity_id != activity_id:       # stopped while we waited
                return None
            if self._mode not in (Mode.IDLE, Mode.FINDING, Mode.SCANNING):
                return None                            # busy listening/speaking/reading
            if activity is Mode.FINDING:
                message = self._process_find(result)
            else:
                message = self._process_scan(result)
        if message:
            self._say(message)
        return message

    def _note_vision_error(self) -> Optional[str]:
        """One camera error is ignored; several in a row end the mode."""
        self._vision_errors += 1
        if self._vision_errors >= self._settings.max_vision_errors:
            self._end_activity()
            return rb.CAMERA_STOPPED_MESSAGE
        return None

    def _process_find(self, result) -> Optional[str]:
        """Find Mode: announce when the object is found, moves, or is lost."""
        result = normalize_find_result(result, self._find_target)
        if result.get("error"):
            return self._note_vision_error()
        self._vision_errors = 0
        observed = None                                # None = not currently seen
        if rb.is_confident_detection(result):
            observed = result.get("position") or "unknown"
        previous = self._find_state
        self._find_state, self._find_pending, changed = _debounced(
            self._find_state, self._find_pending, observed, self._settings.confirm_ticks)
        if not changed:
            return None
        target = self._find_target
        if observed is None:
            return rb.find_lost_message(target)
        self._find_seen = True
        if previous is None:                           # first time (or found again)
            return rb.build_response({"intent": "FIND_OBJECT", "target": target}, result)
        return rb.find_moved_message(target, observed)

    def _process_scan(self, result) -> Optional[str]:
        """Scan Mode: describe the scene once, then only meaningful changes."""
        result = normalize_scene_result(result)
        if not isinstance(result, list):
            return self._note_vision_error()
        self._vision_errors = 0
        snapshot = scene_counts(result)
        if self._scene is None:                        # first look: say what is there
            self._scene = snapshot
            return rb.build_response({"intent": "DESCRIBE_SCENE", "target": None}, result)
        previous = self._scene
        self._scene, self._scene_pending, changed = _debounced(
            self._scene, self._scene_pending, snapshot, self._settings.confirm_ticks)
        if not changed:
            return None
        positions = {}
        for detection in result:
            if rb.is_confident_detection(detection):
                positions.setdefault(label_of(detection), detection.get("position"))
        return rb.scan_changes_message(previous, snapshot, positions) or None
