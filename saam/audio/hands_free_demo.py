"""Offline hands-free voice interface for SAAMOS."""

import os
import queue
import re
import sys
import threading
import time

os.environ["SAAM_DISABLE_SLM"] = "1"

from saam.audio.microphone import record_audio
from saam.audio.stt import load_model, speech_to_text
from saam.audio.tts import speak
from saam.core.modes import ModeController, ModeSettings
from saam.vision.service import VisionService


def normalize_command(text):
    text = " ".join(text.lower().split())
    return re.sub(
        r"^(?:(?:hey|okay|ok)\s+)?"
        r"(?:saam|sam|saamos)[,\s]+",
        "", text
    )


def main():
    print("Loading offline speech recognition...")
    model = load_model()
    vision = VisionService()

    commands = queue.Queue(maxsize=3)
    shutting_down = threading.Event()
    speaking = threading.Event()

    epoch_lock = threading.Lock()
    epoch = [0]

    def say_response(text):
        speaking.set()

        with epoch_lock:
            epoch[0] += 1

        print("\nSAAMOS response:", text, flush=True)

        try:
            speak(text)
        finally:
            with epoch_lock:
                epoch[0] += 1
            speaking.clear()

    controller = ModeController(
        vision=vision,
        speak=say_response,
        start_listening=lambda: None,
        settings=ModeSettings(confirm_ticks=3),
    )

    def microphone_worker():
        device = 1 if sys.platform.startswith("linux") else None

        while not shutting_down.is_set():
            if speaking.is_set():
                time.sleep(0.15)
                continue

            with epoch_lock:
                before = epoch[0]

            try:
                audio = record_audio(duration=3, device=device)
                text = speech_to_text(audio, model)

            except Exception as error:
                print("Microphone error:", error, flush=True)
                time.sleep(1)
                continue

            with epoch_lock:
                overlapped = before != epoch[0]

            if text and not overlapped and not speaking.is_set():
                try:
                    commands.put_nowait(text)
                except queue.Full:
                    print("Skipping extra voice command.")

    print("SAAMOS hands-free mode ready!")
    say_response("SAAMOS is ready. You can speak now.")

    worker = threading.Thread(
        target=microphone_worker,
        daemon=True
    )
    worker.start()

    last_find_update = time.monotonic()

    try:
        while True:
            try:
                heard = commands.get(timeout=0.5)
            except queue.Empty:
                heard = ""

            if heard:
                print("\nSAAMOS heard:", heard, flush=True)
                command = normalize_command(heard)

                if command in {
                    "shut down",
                    "shut down saam",
                    "exit saamos"
                }:
                    break

                if command in {
                    "stop",
                    "stop scanning",
                    "stop searching",
                    "cancel"
                }:
                    controller.on_button_event("long_press")

                elif command in {
                    "repeat",
                    "repeat that",
                    "say that again"
                }:
                    controller.on_button_event("repeat")

                else:
                    if controller.mode.value.lower() == "idle":
                        controller.on_button_event("short_press")
                    controller.handle_text(command)

            controller.tick()

            mode_name = controller.mode.value.lower()
            now = time.monotonic()

            if (
                "find" in mode_name
                and getattr(controller, "_find_state", None) is None
            ):
                if now - last_find_update >= 15:
                    say_response("I'm still searching.")
                    last_find_update = time.monotonic()
            else:
                last_find_update = now

    except KeyboardInterrupt:
        pass

    finally:
        shutting_down.set()
        controller.stop()
        vision.close()
        worker.join(timeout=4)
        print("SAAMOS shut down safely.")


if __name__ == "__main__":
    main()
