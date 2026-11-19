
# SAAMOS - Audio Feedback Tests
# These tests make sure our audio cues work properly.
# They don't need the TI board or actual speakers.

import wave
from pathlib import Path

import pytest

from saam.audio import feedback


def test_all_cues_generate_audio():
    # Make sure every sound creates audio samples.
    for name, pattern in feedback.CUES.items():
        assert len(pattern) > 0

        for frequency, duration, pan in pattern:
            sound = feedback.make_tone(
                frequency, duration, pan
            )

            # Each stereo sample uses 4 bytes total.
            expected = int(
                feedback.SAMPLE_RATE * duration
            ) * 4

            assert len(sound) == expected
            assert any(sound)


def test_invalid_cue():
    # SAAMOS should reject sounds we haven't defined.
    with pytest.raises(ValueError):
        feedback.play_cue("not_a_real_sound")


def test_audio_file_format(monkeypatch):
    # Instead of actually playing audio,
    # we'll inspect the WAV file our code creates.

    checked = []

    def fake_player(command, check):
        # This replaces the audio player during testing.
        assert check is True

        filename = command[-1]

        with wave.open(filename, "rb") as audio:
            assert audio.getnchannels() == 2
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 16000
            assert audio.getnframes() > 0

        checked.append(filename)

    # Temporarily replace subprocess.run.
    monkeypatch.setattr(
        feedback.subprocess, "run", fake_player
    )

    # Test every available sound.
    for name in feedback.CUES:
        feedback.play_cue(name)

    assert len(checked) == len(feedback.CUES)

    # Our temporary audio files should be deleted.
    for filename in checked:
        assert not Path(filename).exists()
