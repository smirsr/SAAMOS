
# SAAMOS - Integrated Audio and Controller Tests
#
# I made this test to check that my audio code can
# communicate with Ari's controller correctly.
#
# We already tested the microphone and Vosk separately.
# Now I want to make sure the entire program connects
# those parts together and stops when the user says stop.
#
# I'm using fake audio and speech recognition here
# so this test doesn't need a microphone or speakers.


from saam.audio import integrated_demo


def test_integrated_audio_demo(monkeypatch):

    # These lists help us keep track of what happens.
    # One stores the responses SAAMOS tries to speak.
    # The other stores the commands sent to the controller.
    spoken_responses = []
    received_commands = []

    # We'll pretend Vosk recognized these two commands.
    # First, the user asks about a bottle.
    # Then, they ask SAAMOS to stop.
    fake_commands = iter([
        "where is my water bottle",
        "stop"
    ])

    # We don't need a real Vosk model for this test,
    # so I'm using a placeholder object.
    fake_model = object()

    # Replace model loading with our placeholder.
    monkeypatch.setattr(
        integrated_demo,
        "load_model",
        lambda: fake_model
    )

    # Replace the microphone recording with fake data.
    # The actual contents don't matter here because
    # we're also replacing speech recognition.
    monkeypatch.setattr(
        integrated_demo,
        "record_audio",
        lambda duration: b"fake audio"
    )

    # Instead of running Vosk, return the next command
    # from our list of pretend speech recognition results.
    def fake_speech_to_text(audio, model):
        assert model is fake_model
        return next(fake_commands)

    monkeypatch.setattr(
        integrated_demo,
        "speech_to_text",
        fake_speech_to_text
    )

    # Keep the real controller so this test checks
    # that the audio demo can actually use Ari's code.
    real_controller = integrated_demo.handle_command

    def check_controller(text):
        received_commands.append(text)
        return real_controller(text)

    monkeypatch.setattr(
        integrated_demo,
        "handle_command",
        check_controller
    )

    # Replace the speaker function so the test doesn't
    # actually make my computer say anything out loud.
    # Instead, it saves the sentences in a list.
    monkeypatch.setattr(
        integrated_demo,
        "speak",
        lambda text: spoken_responses.append(text)
    )

    # Run the integrated audio demonstration.
    # It should process the bottle command and then stop.
    integrated_demo.run_integrated_demo()

    # Check that both commands reached the controller.
    assert received_commands == [
        "where is my water bottle",
        "stop"
    ]

    # Check that the controller's simulated bottle
    # location was passed to our speech output.
    assert "Your bottle is on the left." in spoken_responses

    # Check that the STOP response was also spoken.
    assert "Okay, stopping." in spoken_responses

    # If the loop doesn't stop after the second command,
    # our fake command list will run out and the test fails.
