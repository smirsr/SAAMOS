
# SAAMOS - Integrated Audio and Controller Demo
#
# This file connects my audio modules with Ari's controller.
# My audio code handles listening and speaking, while
# the controller figures out what the user is asking.
#
# We are using mock vision for now since we haven't
# connected the real camera module yet.
#
# The goal is to test the full voice-command process
# before running everything on the TI board.


# This lets us turn off the optional SLM for this demo.
# We want to use the rule-based parser for now because
# it doesn't need a separate language model.
import os

# Bring in the microphone code I already tested.
from saam.audio.microphone import record_audio

# Bring in our offline Vosk speech recognition.
from saam.audio.stt import load_model, speech_to_text

# Bring in the text-to-speech function.
from saam.audio.tts import speak

# Import Ari's controller and intent constants.
from saam.core.controller import handle_command
from saam.core.intent_parser import STOP, parse_intent


def run_integrated_demo():

    # For our first prototype, we are disabling the SLM.
    # This makes sure the controller uses its simpler
    # rule-based parser instead.
    os.environ["SAAM_DISABLE_SLM"] = "1"

    # Load Vosk once before starting the listening loop.
    # We don't want to reload the model every time we
    # record another command.
    model = load_model()

    print("SAAMOS integrated demo is ready.")
    print("DEMO MODE: Using simulated vision results.")

    # Let the user know that the system has started.
    speak("SAAMOS is ready. Vision is in demonstration mode.")

    # Continue recording commands until the user says stop.
    while True:

        # Record five seconds of microphone audio.
        audio = record_audio(duration=5)

        # Convert the recorded audio into text using Vosk.
        text = speech_to_text(audio, model)

        print("SAAMOS heard:", text)

        # If nothing was recognized, skip this recording.
        if not text:
            print("No speech recognized. Trying again...")
            continue

        # Send the recognized text to Ari's controller.
        #
        # The controller checks the command, uses mock vision
        # when necessary, and creates a response sentence.
        response = handle_command(text)

        print("SAAMOS response:", response)

        # Speak the response using our TTS module.
        speak(response)

        # Check whether the user asked SAAMOS to stop.
        #
        # We use the same intent parser as the controller
        # so the stop decision matches its command logic.
        intent = parse_intent(text)

        if intent["intent"] == STOP:
            print("SAAMOS is shutting down.")
            break


if __name__ == "__main__":
    run_integrated_demo()
