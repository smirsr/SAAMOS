import subprocess
import sys


def speak(text: str) -> None:
    """Speak text using the macOS speech engine."""
    if sys.platform != "darwin":
        raise RuntimeError(
            "This initial TTS implementation only supports macOS."
        )

    subprocess.run(["say", text], check=True)


if __name__ == "__main__":
    speak("SAAMOS is ready to assist you.")