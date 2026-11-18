
# SAAMOS - Audio Feedback
# This file makes short sounds to tell the user what SAAMOS is doing.
# We can test these on my Mac before moving them to the TI board.

import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

# How many audio samples we generate every second.
SAMPLE_RATE = 16000

# These are the sound patterns we want SAAMOS to use.
# Each part has a frequency, duration, and left/right position.
# Pan: -1 means left, 0 means center, 1 means right.
CUES = {
    "listening": [(700, 0.10, 0), (950, 0.10, 0)],
    "left": [(450, 0.14, -1), (450, 0.14, -1)],
    "center": [(650, 0.25, 0)],
    "right": [(900, 0.14, 1), (900, 0.14, 1)],
    "error": [(300, 0.18, 0), (220, 0.22, 0)],
}


def make_tone(frequency, duration, pan=0):
    # Generate a sine wave. This makes a smooth beep sound.
    frames = bytearray()
    total = int(SAMPLE_RATE * duration)

    # Keep some sound on both channels so neither side goes silent.
    left_gain = 0.65 - 0.35 * pan
    right_gain = 0.65 + 0.35 * pan

    for i in range(total):
        time_position = i / SAMPLE_RATE

        # Fade the beginning and end to avoid clicking noises.
        fade = min(1.0, i / 128, (total - i - 1) / 128)

        sample = int(
            9000 * fade *
            math.sin(2 * math.pi * frequency * time_position)
        )

        # Create separate left and right speaker samples.
        left = int(sample * left_gain)
        right = int(sample * right_gain)

        frames.extend(struct.pack("<hh", left, right))

    return frames


def play_cue(name):
    # Make sure this is one of the sounds we defined.
    if name not in CUES:
        raise ValueError(f"Unknown SAAMOS audio cue: {name}")

    # Temporary WAV file. We'll delete it after playing.
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as temp:
        filename = temp.name

    try:
        with wave.open(filename, "wb") as audio:
            audio.setnchannels(2)   # Stereo: left and right
            audio.setsampwidth(2)   # 16-bit audio
            audio.setframerate(SAMPLE_RATE)

            for frequency, duration, pan in CUES[name]:
                audio.writeframes(
                    make_tone(frequency, duration, pan)
                )

                # Add a short silence between beeps.
                silence = b"\x00\x00\x00\x00" * 960
                audio.writeframes(silence)

        # The Mac and TI board use different audio players.
        if sys.platform == "darwin":
            command = ["afplay", filename]

        elif sys.platform.startswith("linux"):
            command = ["aplay", "-q"]

            # Optional audio device, selected during board testing.
            device = os.environ.get("SAAM_AUDIO_DEVICE")
            if device:
                command.extend(["-D", device])

            command.append(filename)

        else:
            raise RuntimeError("Audio player not configured")

        # Wait until the beep finishes before continuing.
        subprocess.run(command, check=True)

    finally:
        # Clean up even if the audio player encounters an error.
        os.remove(filename)
