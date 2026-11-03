
# SAAMOS - Full Audio + Real Vision Demo
#
# This connects the audio system I worked on,
# Ari's command controller, and our team's
# actual object detection system.
#
# The microphone listens to a command.
# Vosk converts it into text.
# The controller decides what the command means.
# Vision uses the real camera when needed.
# Finally, SAAMOS speaks the response.
#
# This is running locally without an internet connection.
# The optional SLM is disabled for our first prototype.

import os

from saam.audio.microphone import record_audio
from saam.audio.stt import load_model, speech_to_text
from saam.audio.tts import speak

from saam.core.controller import handle_command
from saam.core.intent_parser import parse_intent, STOP

from saam.vision.service import VisionService


def main():

    # Keep the optional SLM disabled for this demo.
    # This lets us use Ari's rule-based parser.
    os.environ["SAAM_DISABLE_SLM"] = "1"

    # Load Vosk once so it doesn't reload
    # every time we record another command.
    model = load_model()

    # Set up the real vision service.
    # This uses the camera chosen by SAAM_CAMERA.
    vision = VisionService()

    print("SAAMOS is ready!")
    print("Using real object detection.")

    speak("SAAMOS is ready.")

    try:
        while True:

            # Record five seconds from the microphone.
            audio = record_audio(duration=5)

            # Turn the recording into recognized words.
            text = speech_to_text(audio, model)

            print("SAAMOS heard:", text)

            # If nothing was recognized, try again.
            if not text:
                continue

            # Send the recognized words to the controller.
            # These callbacks tell the controller to use
            # the real camera instead of MockVision.
            response = handle_command(
                text,
                detect_object=vision.detect_object,
                describe_scene=vision.describe_scene
            )

            print("SAAMOS response:", response)

            # Speak the controller's response out loud.
            speak(response)

            # Stop the loop when the user says stop.
            intent = parse_intent(text)

            if intent["intent"] == STOP:
                print("SAAMOS is shutting down.")
                break

    finally:
        # Always release the camera properly.
        # This also runs if the program is interrupted.
        vision.close()


if __name__ == "__main__":
    main()
