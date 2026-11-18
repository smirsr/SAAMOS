
# SAAMOS - Button Tests
# These tests check our button detection logic.
# We don't need the TI board connected for these tests.

from saam.hardware.button import ButtonListener


def test_first_reading():
    # The first reading sets our starting point.
    # It shouldn't count as a new button event.
    button = ButtonListener(lambda: 10)

    assert button.poll() is False


def test_button_event():
    # Pretend our hardware counter increases from 10 to 11.
    counts = iter([10, 11])

    button = ButtonListener(
        lambda: next(counts),
        cooldown_seconds=0
    )

    assert button.poll() is False
    assert button.poll() is True


def test_no_new_event():
    # If the counter stays the same,
    # SAAMOS shouldn't detect another event.
    counts = iter([10, 10, 10])

    button = ButtonListener(
        lambda: next(counts)
    )

    assert button.poll() is False
    assert button.poll() is False
    assert button.poll() is False


def test_cooldown():
    # We want to prevent events that happen too quickly.
    # A long cooldown lets us test this without sleeping.
    counts = iter([10, 11, 12])

    button = ButtonListener(
        lambda: next(counts),
        cooldown_seconds=1000
    )

    assert button.poll() is False

    # First increase should work.
    assert button.poll() is True

    # Second increase should be ignored.
    assert button.poll() is False
