"""Keyboard demo of operating modes and push-to-talk (mock vision, no hardware).

Run from the repository root:

    python -m saam.core.modes_demo

Type one line at a time:

    <any sentence>        a voice command, e.g.  Find my bottle
    !short  !long  !repeat   button events (or the keys s, l, r)
    .                     one tick (the host loop calls tick() about twice a second)
    .5                    five ticks in a row
    /add cup left         add an object to the mock scene (left, center, right)
    /remove cup           remove an object from the mock scene
    /move bottle right    move an object
    /mode                 show the current mode
    /quit                 leave

Everything SAAMOS "says" is printed as  [SAAMOS] ...
This is a separate demo; the original one (python -m saam.core.controller)
is untouched.
"""

import re

from .controller import MockVision
from .modes import ModeController, ModeSettings


def main() -> None:
    vision = MockVision()
    controller = ModeController(
        vision=vision,
        speak=lambda text: print("[SAAMOS] " + text),
        start_listening=lambda: print("[mic] listening... type your command"),
        stop_speaking=lambda: print("[audio] speech cut off"),
        settings=ModeSettings(confirm_ticks=2),
    )
    print(__doc__)
    print("Mode: {}".format(controller.mode.value))
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line in ("/quit", "/q"):
            break
        if line == "/mode":
            print("Mode: {} (activity: {})".format(
                controller.mode.value, controller.activity.value if controller.activity else "none"))
            continue
        parts = line.split()
        if line.startswith("!"):
            controller.on_button_event(line[1:])
        elif line in ("s", "l", "r"):
            controller.on_key(line)
        elif parts and parts[0] == "/add" and len(parts) >= 2:
            vision.add(parts[1], parts[2] if len(parts) > 2 else "center")
        elif parts and parts[0] == "/remove" and len(parts) == 2:
            vision.remove(parts[1])
        elif parts and parts[0] == "/move" and len(parts) == 3:
            vision.set_position(parts[1], parts[2])
        elif re.fullmatch(r"\.\d*", line):
            for _ in range(int(line[1:] or 1)):
                controller.tick()
        elif line:
            controller.handle_text(line)
        print("Mode: {}".format(controller.mode.value))
    controller.stop()


if __name__ == "__main__":
    main()
