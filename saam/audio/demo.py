
# SAAMOS - Audio Demo
#
# This file connects the different audio features I worked on.
# Instead of recording, recognizing, and speaking separately,
# I want SAAMOS to be able to do all three together.
#
# For now, I am using a few basic voice commands.
# This lets us test the full audio system before we connect
# it to the rest of the project.


# Import our microphone recording function.
# This is the part that actually listens to the user.
from saam.audio.microphone import record_audio

# Import the speech recognition functions.
# These let us load our offline model and turn audio into text.
from saam.audio.stt import load_model, speech_to_text

# Import the text-to-speech function we already created.
# This is what makes SAAMOS speak its response out loud.
from saam.audio.tts import speak


def process_command(text):

    # Convert the recognized text to lowercase.
    # This makes it easier to check commands without
    # worrying about capital letters.
    text = text.lower().strip()

    # Check for a stop command first.
    # We return True as the second value so the program
    # knows it should stop listening after this response.
    if "stop" in text or "goodbye" in text:
        return "Goodbye! SAAMOS is shutting down.", True

    # If the user says hello, give a simple greeting.
    elif "hello" in text or "hi saamos" in text:
        return "Hello! I'm SAAMOS, your offline assistant.", False

    # Explain what the system can currently do.
    elif "what can you do" in text:
        return (
            "I can listen to commands and respond "
            "using offline speech recognition."
        ), False

    # Give the user a list of supported commands.
    elif "help" in text:
        return (
            "You can say hello, ask what I can do, "
            "or tell me to stop."
        ), False

    # If the speech was recognized but we don't have
    # a matching command, explain that to the user.
    else:
        return (
            "Sorry, I don't recognize that command yet."
        ), False


def run_demo():

    # Load the offline Vosk model once at the beginning.
    # We do not want to reload it after every command
    # because loading the model takes extra time.
    model = load_model()

    print("SAAMOS audio system is ready!")

    # This makes SAAMOS speak when the demo first starts.
    speak("SAAMOS audio system is ready.")

    # Keep the program running until the user says stop.
    while True:

        # Record five seconds of microphone audio.
        audio = record_audio(duration=7)

        # Convert the recorded audio into recognized text.
        text = speech_to_text(audio, model)

        print("SAAMOS heard:", text)

        # If Vosk couldn't recognize anything,
        # skip processing and start a new recording.
        if not text:
            print("No speech recognized. Trying again...")
            continue

        # Figure out which command the user gave.
        response, should_stop = process_command(text)

        # Show the response in the terminal as well.
        print("SAAMOS response:", response)

        # Have SAAMOS speak the response out loud.
        speak(response)

        # If the user said stop, end the loop.
        if should_stop:
            break

    print("SAAMOS audio demo finished.")


# This only runs when we execute demo.py directly.
# It will not automatically start when another file
# imports something from this module.
if __name__ == "__main__":
    run_demo()
