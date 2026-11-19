
# SAAMOS - Automatic Find and Scan Demo
#
# This version updates Find and Scan modes automatically.
# I don't have to keep pressing "." to check the camera.
#
# We are keeping the existing working demos unchanged.

import os
import select
import sys
import time

# Use our offline rule-based command parser.
os.environ["SAAM_DISABLE_SLM"] = "1"

from saam.audio.microphone import record_audio
from saam.audio.stt import load_model, speech_to_text
from saam.audio.tts import speak
from saam.audio.feedback import play_cue

from saam.core.modes import ModeController, ModeSettings
from saam.vision.service import VisionService


def main():
    print("Loading SAAMOS...")

    # Load the speech recognition model only once.
    model = load_model()

    # Open the real camera and object detector.
    vision = VisionService()

    def say_response(text):
        # Show what SAAMOS says in the terminal.
        print("\nSAAMOS response:", text)

        # Speak the same response out loud.
        speak(text)

    def listen_once():
        # Play our short listening sound.
        play_cue("listening")

        print("\nSAAMOS is listening...")

        try:
            # On the TI board, device 1 is our USB webcam microphone.
            # Keep the Mac's default microphone for local testing.
            microphone_device = 1 if sys.platform.startswith("linux") else None
            audio = record_audio(duration=5, device=microphone_device)

            # Convert the recorded audio into words.
            text = speech_to_text(audio, model)

        except Exception as error:
            print("Audio error:", error)
            play_cue("error")
            controller.stop()
            return

        print("SAAMOS heard:", text)

        if not text:
            print("No speech recognized.")
            play_cue("error")
            controller.stop()
            return

        # Give Ari's controller the recognized command.
        controller.handle_text(text)

    # Connect our real components to Ari's controller.
    controller = ModeController(
        vision=vision,
        speak=say_response,
        start_listening=listen_once,
        settings=ModeSettings(confirm_ticks=3),
    )
    # Track when we last gave a search progress update.

    last_find_update = time.monotonic()

    print("\nSAAMOS is ready!")
    print("Find and Scan now update automatically.")
    print("s = listen, l = stop, r = repeat, q = quit")

    try:
        print("\nSAAMOS > ", end="", flush=True)

        while True:
            # Check whether the user typed a command.
            #
            # Instead of waiting forever for input(),
            # select waits up to 0.5 seconds.
            #
            # If nobody types anything, we use that time
            # to update the camera and controller.
            ready, _, _ = select.select(
                [sys.stdin], [], [], 0.5
            )

            if not ready:
                # Automatically update Find or Scan Mode.
                controller.tick()

                mode_name = getattr(
                    controller.mode, "value", str(controller.mode)
                ).lower()
                now = time.monotonic()

                if (
                    "find" in mode_name
                    and getattr(controller, "_find_state", None) is None
                ):
                    if now - last_find_update >= 15:
                        say_response("I'm still searching.")
                        last_find_update = now
                else:
                    last_find_update = now

                continue

            # Read the command entered in the terminal.
            line = sys.stdin.readline()

            if not line:
                break

            command = line.strip().lower()

            if command == "q":
                break

            elif command == "s":
                controller.on_button_event("short_press")

            elif command == "l":
                controller.on_button_event("long_press")

            elif command == "r":
                controller.on_button_event("repeat")

            elif command == ".":
                # Optional manual camera check.
                controller.tick()

            elif command:
                # Allow typed commands while debugging.
                controller.handle_text(command)

            print("Current mode:", controller.mode.value)
            print("SAAMOS > ", end="", flush=True)

    finally:
        # Stop all activities and close the camera.
        controller.stop()
        vision.close()
        print("\nSAAMOS shut down safely.")


if __name__ == "__main__":
    main()
