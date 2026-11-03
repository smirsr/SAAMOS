from unittest.mock import patch

from saam.audio.tts import speak


@patch("saam.audio.tts.subprocess.run")
@patch("saam.audio.tts.sys.platform", "darwin")
def test_speak(mock_run):
    speak("Object detected on your left.")

    mock_run.assert_called_once_with(
        ["say", "Object detected on your left."],
        check=True,
    )


# This test checks our Linux speech command.
# I am using monkeypatch so we don't actually need
# to run Linux or have eSpeak NG installed on my Mac.

def test_linux_speak(monkeypatch):

    # Import the complete module so we can temporarily
    # change its operating system and speech command.
    from saam.audio import tts

    # This list will store the command our function tries
    # to run, so we can check whether it is correct.
    commands = []

    # Pretend the computer is running Linux.
    monkeypatch.setattr(tts.sys, "platform", "linux")

    # Pretend eSpeak NG is installed.
    monkeypatch.setattr(
        tts.shutil,
        "which",
        lambda name: "/usr/bin/espeak-ng"
    )

    # Replace subprocess.run with a fake version so
    # the test doesn't actually try to play audio.
    def fake_run(command, check):
        commands.append(command)

    monkeypatch.setattr(tts.subprocess, "run", fake_run)

    # Ask our function to speak a test sentence.
    tts.speak("Hello from SAAMOS")

    # Check that the correct Linux command was selected.
    assert commands == [
        ["espeak-ng", "-s", "150", "Hello from SAAMOS"]
    ]
