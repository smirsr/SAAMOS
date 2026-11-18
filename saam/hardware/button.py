
# SAAMOS - Button Input
# This file handles button-related events.
#
# For now, we are testing with a counter that we can simulate.
# Later, we will connect it to the TI board's button input.
#
# The actual SW4 interrupt counter is shared with other hardware,
# so a counter change alone does not guarantee a button press.

import time


class ButtonListener:
    def __init__(self, read_counter, cooldown_seconds=0.5):
        # This function tells us the current counter value.
        self.read_counter = read_counter

        # Prevents events from happening too close together.
        self.cooldown_seconds = cooldown_seconds

        # Store the previous counter value so we can compare.
        self.last_count = None

        # Remember when we last accepted a button event.
        self.last_press_time = None

    def poll(self):
        # Read the latest counter value.
        current_count = self.read_counter()

        # The first reading is just our starting point.
        if self.last_count is None:
            self.last_count = current_count
            return False

        # Check whether the counter increased.
        changed = current_count > self.last_count

        # Save this reading for next time.
        self.last_count = current_count

        if not changed:
            return False

        # Check how long it has been since the last event.
        now = time.monotonic()

        if self.last_press_time is not None:
            elapsed = now - self.last_press_time

            if elapsed < self.cooldown_seconds:
                return False

        # We detected an accepted counter change.
        self.last_press_time = now
        return True
