
# SAAMOS - Microphone Tests
#
# I made these tests to check that our microphone function
# records the correct amount of audio and returns it in
# a format that we can use for speech recognition.
#
# Instead of using the actual microphone every time,
# we use mock audio so our tests can run automatically.


# pytest helps us run and organize our tests.
import pytest

# numpy lets us create fake audio samples for testing.
import numpy as np

# We import our microphone module so we can test its functions.
from saam.audio import microphone


def test_record_audio(monkeypatch):

    # We are testing a one-second recording.
    # At 16000 samples per second, this should give us
    # exactly 16000 audio samples.
    duration = 1
    expected_samples = duration * microphone.SAMPLE_RATE

    # Create fake microphone data.
    # We use zeros because we only care about the recording
    # format and length for this particular test.
    fake_audio = np.zeros(
        (expected_samples, 1),
        dtype=np.int16
    )

    # This function replaces the real microphone recording.
    # It accepts the same arguments as sounddevice.rec().
    def fake_record(frames, samplerate, channels, dtype, device):

        # Check that our recording function is asking
        # for the settings that Vosk will need later.
        assert frames == expected_samples
        assert samplerate == 16000
        assert channels == 1
        assert dtype == "int16"

        return fake_audio

    # monkeypatch temporarily replaces a real function
    # with our fake function just for this test.
    #
    # This way, the test does not need microphone permission
    # or any actual recording hardware.
    monkeypatch.setattr(microphone.sd, "rec", fake_record)

    # We also replace sd.wait() because there is no
    # real recording that we need to wait for.
    monkeypatch.setattr(microphone.sd, "wait", lambda: None)

    # Call our actual SAAMOS recording function.
    recording = microphone.record_audio(duration=duration)

    # Make sure the total number of samples is correct.
    assert len(recording) == expected_samples

    # Make sure audio is stored using 16-bit integers.
    assert recording.dtype == np.int16

    # Make sure we returned a one-dimensional array.
    assert recording.ndim == 1