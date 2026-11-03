
# SAAMOS - Voice Command Tests
#
# I want to make sure SAAMOS responds correctly
# when it receives different voice commands.
#
# Since Vosk sometimes recognizes words incorrectly,
# I'm testing the command processing separately.
# This way, I can check if the problem is with my
# Python code or with the speech recognition itself.


# Import the command function from our audio demo.
from saam.audio.demo import process_command


def test_hello_command():

    # Check what happens when the user says hello.
    response, should_stop = process_command("hello saamos")

    # SAAMOS should return its greeting.
    assert response == "Hello! I'm SAAMOS, your offline assistant."

    # Saying hello should not end the program.
    assert should_stop is False


def test_help_command():

    # Check that the help command works when the word
    # is actually recognized correctly.
    response, should_stop = process_command("help")

    # The response should explain available commands.
    assert "You can say hello" in response

    # Asking for help should keep SAAMOS running.
    assert should_stop is False


def test_capabilities_command():

    # Check that SAAMOS can explain what it does.
    response, should_stop = process_command("what can you do")

    assert "offline speech recognition" in response
    assert should_stop is False


def test_stop_command():

    # When the user says stop, SAAMOS should exit
    # the listening loop after giving its response.
    response, should_stop = process_command("stop")

    assert "Goodbye" in response
    assert should_stop is True


def test_unknown_command():

    # Try a sentence that we have not programmed
    # SAAMOS to understand yet.
    response, should_stop = process_command("turn on the lights")

    # It should explain that the command is unsupported.
    assert "don't recognize" in response

    # An unknown command should not shut down SAAMOS.
    assert should_stop is False


def test_uppercase_command():

    # Make sure our lowercase conversion works.
    # The user might type a command using capital letters,
    # even though Vosk usually returns lowercase text.
    response, should_stop = process_command("HELP")

    assert "You can say hello" in response
    assert should_stop is False
