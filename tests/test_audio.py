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