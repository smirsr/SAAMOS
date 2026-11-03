
# SAAMOS - Text-to-Speech
#
# This file lets SAAMOS speak responses out loud.
# I originally used the Mac's built-in say command,
# but our final project is supposed to run on a TI board.
#
# Since the TI board uses Linux, I added support for
# eSpeak NG so we can use the same Python function
# on both my Mac and the board.


import subprocess
import sys
import shutil


def speak(text: str) -> None:

    # First, check which operating system we're using.
    # sys.platform returns "darwin" on macOS and
    # "linux" on most Linux systems.
    if sys.platform == "darwin":

        # My Mac already has the say command built in,
        # so I can use it without installing anything else.
        command = ["say", text]

    elif sys.platform.startswith("linux"):

        # On Linux, we want to use eSpeak NG.
        # shutil.which checks if the command is installed
        # and available in our system's PATH.
        if shutil.which("espeak-ng") is None:
            raise RuntimeError(
                "eSpeak NG is not installed on this Linux system."
            )

        # -s 150 sets the speaking speed to 150 words/minute.
        # This gives us a slightly slower spoken response.
        command = ["espeak-ng", "-s", "150", text]

    else:

        # We haven't added support for other systems yet.
        raise RuntimeError(
            f"Unsupported operating system: {sys.platform}"
        )

    # Run the command and wait until it finishes.
    #
    # We use a list instead of one long command string.
    # This also avoids passing the user's text through a shell.
    #
    # check=True makes Python raise an error if the
    # speech command fails.
    subprocess.run(command, check=True)


# Run this test only when we execute this file directly.
if __name__ == "__main__":
    speak("SAAMOS is ready to assist you.")
