
# SAAMOS - Microphone Capture
#
# This file handles recording audio from a microphone.
# We need this because SAAMOS has to be able to hear the user
# before we can convert their speech into text.
#
# For now, we are recording a few seconds at a time.
# Later, we can connect this to our speech-to-text module
# so SAAMOS can recognize what the user is saying.


# sounddevice lets Python communicate with audio devices.
# We use it to record sound from our microphone.
import sounddevice as sd

# numpy is used to work with the audio samples.
# The recording is basically a collection of numbers
# representing the sound picked up by the microphone.
import numpy as np


# We are using 16000 samples per second.
# This is a common sample rate for speech recognition.
SAMPLE_RATE = 16000

# We will record for 5 seconds during our initial tests.
RECORDING_SECONDS = 5


def record_audio(duration=RECORDING_SECONDS, device=None):
    """
    Records audio from a microphone and returns the samples.

    duration: how many seconds we want to record
    device: which microphone to use (None uses the default)
    """

    print("SAAMOS is listening...")

    # We multiply the recording time by the sample rate
    # to figure out how many samples need to be recorded.
    #
    # For example:
    # 5 seconds * 16000 samples/second = 80000 samples.
    total_samples = int(duration * SAMPLE_RATE)

    # This starts recording from our selected microphone.
    #
    # samplerate: number of audio samples collected per second
    # channels=1: records mono audio instead of stereo
    # dtype="int16": stores samples as 16-bit integers
    #
    # Mono 16-bit PCM audio is a useful format for Vosk.
    audio = sd.rec(
        total_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device
    )

    # The recording function starts capturing audio.
    # sd.wait() makes the program wait until recording finishes.
    sd.wait()

    print("Recording finished!")

    # Flatten turns the audio array into a one-dimensional
    # list of samples, which makes later processing easier.
    return audio.flatten()


if __name__ == "__main__":

    # This section only runs when we execute this file directly.
    # It lets us test the microphone without running all of SAAMOS.

    recording = record_audio()

    # Show how many samples we collected.
    print("Total audio samples:", len(recording))

    # Show the recording length in seconds.
    print("Recording duration:", len(recording) / SAMPLE_RATE)

    # Check whether the audio contains nonzero samples.
    # This doesn't prove speech was recorded, but it helps us
    # catch cases where the microphone recorded only silence.
    if np.any(recording != 0):
        print("Microphone captured audio data!")
    else:
        print("Warning: The recording contains only zeros.")
