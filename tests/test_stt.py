
# SAAMOS - Speech-to-Text Tests
#
# These tests check that our speech recognition code
# can process audio data and return recognized words.
#
# We are using fake recognition results so we can test
# the Python logic without needing to speak every time.


import json
import numpy as np

# Import our speech-to-text module.
from saam.audio import stt


def test_speech_to_text(monkeypatch):

    # Create a fake recognizer that behaves like Vosk.
    # We want to test how our code handles the result
    # without loading the actual speech model.

    class FakeRecognizer:

        def __init__(self, model, sample_rate):
            # Check that we are using the same sample rate
            # as our microphone recording function.
            assert sample_rate == 16000

        def AcceptWaveform(self, audio_bytes):
            # Make sure the audio was converted into bytes.
            assert isinstance(audio_bytes, bytes)
            return True

        def FinalResult(self):
            # Pretend Vosk recognized this sentence.
            return json.dumps({
                "text": "hello saamos"
            })

    # Replace the real recognizer with our fake one.
    monkeypatch.setattr(stt, "KaldiRecognizer", FakeRecognizer)

    # Create one second of fake audio.
    fake_audio = np.zeros(16000, dtype=np.int16)

    # The model is only a placeholder for this test.
    fake_model = object()

    # Run our actual speech-to-text function.
    result = stt.speech_to_text(fake_audio, fake_model)

    # Check that our function returns the expected words.
    assert result == "hello saamos"
