
from saam.core.scan_stability import debounce_scene_counts


def run_checks(confirmed, frames):
    # Feed camera observations into the filter one by one.
    pending = None
    changes = []

    for frame in frames:
        confirmed, pending, changed = debounce_scene_counts(
            confirmed, pending, frame, needed=3
        )
        changes.append(changed)

    return confirmed, changes


def test_cup_confirmed_despite_flickering_chair():
    # The cup stays visible, but the chair keeps changing.
    scene, changes = run_checks({}, [
        {"cup": 1, "chair": 1},
        {"cup": 1},
        {"cup": 1, "chair": 1},
    ])

    assert scene == {"cup": 1}
    assert changes == [False, False, True]


def test_cup_disappears_despite_background_changes():
    # The cup stays missing while the chair flickers.
    scene, changes = run_checks(
        {"cup": 1, "person": 1},
        [
            {"person": 1, "chair": 1},
            {"person": 1},
            {"person": 1, "chair": 1},
        ]
    )

    assert scene == {"person": 1}
    assert changes == [False, False, True]


def test_single_false_detection_ignored():
    # One random detection should not be announced.
    scene, changes = run_checks({}, [
        {"parking meter": 1},
        {},
        {},
    ])

    assert scene == {}
    assert changes == [False, False, False]


def test_object_count_change_confirmed():
    # Two cups must be seen consistently before updating.
    scene, changes = run_checks({"cup": 1}, [
        {"cup": 2},
        {"cup": 2},
        {"cup": 2},
    ])

    assert scene == {"cup": 2}
    assert changes == [False, False, True]
