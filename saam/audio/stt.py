
# SAAMOS - Offline Speech-to-Text
#
# This file converts microphone recordings into text.
# We are using Vosk because it works locally, meaning
# SAAMOS does not need internet to recognize speech.
#
# The microphone.py file handles recording the audio.
# This file handles figuring out what words were spoken.


# json lets us read the results that Vosk gives us.
# Vosk returns information in JSON format, which is
# basically text organized into keys and values.
import json

# Path helps us locate the model folder without
# depending on which directory we run Python from.
from pathlib import Path

# These are the two main Vosk components we need.
# Model loads the language model from our computer.
# KaldiRecognizer processes audio and recognizes words.
from vosk import Model, KaldiRecognizer, SetLogLevel

# Import our existing microphone code.
# This means we can reuse record_audio() instead of
# writing the microphone recording code again.
from saam.audio.microphone import record_audio, SAMPLE_RATE


# Find the main SAAMOS project directory.
# parents[2] goes from saam/audio/stt.py back to SAAMOS.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# This is the Vosk English model we downloaded earlier.
MODEL_PATH = PROJECT_ROOT / "models" / "vosk-model-small-en-us-0.15"


def load_model():
    # First, check that the model folder actually exists.
    # If it does not, we want a clear error message instead
    # of the program failing without explaining why.
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Could not find the Vosk model at: {MODEL_PATH}"
        )

    # Vosk can print a lot of technical information.
    # Setting the log level to -1 hides most of that output.
    SetLogLevel(-1)

    print("Loading SAAMOS speech recognition model...")

    # Load the downloaded model from our computer.
    # Once installed, this does not require internet.
    model = Model(str(MODEL_PATH))

    print("Speech recognition model loaded!")

    return model


def speech_to_text(audio, model):
    # This function takes audio samples and converts them
    # into words using the Vosk model.

    # The recognizer needs the model and the sample rate.
    # Our microphone is recording at 16000 samples/second.
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    # Vosk processes raw audio bytes.
    # Our microphone returns a numpy array, so we convert
    # that array into bytes before passing it to Vosk.
    audio_bytes = audio.tobytes()

    # Give Vosk the recorded audio.
    # We only need the completed result for this first
    # prototype, so we collect it after all audio is sent.
    recognizer.AcceptWaveform(audio_bytes)

    # FinalResult() returns the recognized speech as JSON.
    result_json = recognizer.FinalResult()

    # Convert the JSON text into a Python dictionary.
    result = json.loads(result_json)

    # Get only the recognized text from the result.
    # If no words were recognized, return an empty string.
    recognized_text = result.get("text", "")

    return recognized_text


if __name__ == "__main__":

    # Load the model once before starting the recording.
    model = load_model()

    # Record five seconds using our microphone module.
    # We are reusing the function we already tested.
    audio = record_audio(duration=5)

    # Convert our recording into text.
    text = speech_to_text(audio, model)

    # Print the recognition result so we can verify it.
    if text:
        print("SAAMOS heard:", text)
    else:
        print("SAAMOS could not recognize any words.")
