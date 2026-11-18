
# SAAMOS - Final Integration Demo
#
# This connects our real microphone, camera, and speaker
# to Ari's new controller.
#
# For now, I can use the keyboard instead of the SW4 button.
# This helps us test the software before changing the TI board.
#
# The original full_demo.py stays untouched.

import os

# We are keeping the optional SLM disabled.
# Commands will use Ari's rule-based parser.
os.environ["SAAM_DISABLE_SLM"] = "1"

from saam.audio.microphone import record_audio
from saam.audio.stt import load_model, speech_to_text
from saam.audio.tts import speak
from saam.audio.feedback import play_cue

from saam.core.modes import ModeController
from saam.vision.service import VisionService


def main():

    print("Loading SAAMOS speech model...")

    # Load Vosk only once when SAAMOS starts.
    model = load_model()

    # Set up our existing real camera system.
    vision = VisionService()

    # This function is used whenever Ari's controller
    # wants SAAMOS to say something.
    def say_response(text):
        print("SAAMOS response:", text)
        speak(text)

    # We define this function before creating the controller.
    # Python will use the controller variable when it is called.
    def listen_once():

        # Let the user know recording is about to begin.
        play_cue("listening")

        print("SAAMOS is listening...")

        try:
            # Record five seconds from the microphone.
            audio = record_audio(duration=5)

            # Use Vosk to turn the recording into text.
            text = speech_to_text(audio, model)

        except Exception as error:
            print("Microphone or speech error:", error)
            play_cue("error")
            controller.stop()
            return

        print("SAAMOS heard:", text)

        # If nothing was recognized, don't process a command.
        if not text:
            print("No speech recognized.")
            play_cue("error")
            controller.stop()
            return

        # Send the recognized words to Ari's controller.
        controller.handle_text(text)

    # Connect our real hardware functions to Ari's controller.
    controller = ModeController(
        vision=vision,
        speak=say_response,
        start_listening=listen_once,
    )

    print("\nSAAMOS is ready!")
    print("Real microphone and vision enabled.")
    print("s = listen, . = update, l = stop")
    print("r = repeat, q = quit\n")

    try:
        while True:

            # Keyboard commands simulate button actions.
            command = input("SAAMOS > ").strip().lower()

            if command == "q":
                break

            elif command == "s":
                # Simulates a short button press.
                controller.on_button_event("short_press")

            elif command == "l":
                # Simulates a long button press.
                controller.on_button_event("long_press")

            elif command == "r":
                # Repeat the last spoken response.
                controller.on_button_event("repeat")

            elif command == ".":
                # Update Find or Scan mode once.
                controller.tick()

            elif command:
                # Also allow typed commands for debugging.
                controller.handle_text(command)

            print("Current mode:", controller.mode.value)

    finally:
        # Clean up when we quit or interrupt the program.
        controller.stop()
        vision.close()
        print("SAAMOS shut down safely.")


if __name__ == "__main__":
    main()
