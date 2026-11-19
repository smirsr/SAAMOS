"""Keep Scan Mode announcements stable across camera frames."""


def debounce_scene_counts(confirmed, pending, observed, needed):
    """
    Confirm changes to each object separately.

    confirmed = the objects SAAMOS already trusts
    pending = object changes we're still checking
    observed = what the camera sees right now
    needed = number of matching checks required
    """

    # Copy the dictionaries so we don't change the old scene.
    updated = dict(confirmed)
    waiting = dict(pending or {})

    # Check every object we've seen or are currently tracking.
    labels = set(confirmed) | set(observed) | set(waiting)

    for label in labels:
        old_count = confirmed.get(label, 0)
        new_count = observed.get(label, 0)

        # Nothing changed for this object.
        if old_count == new_count:
            waiting.pop(label, None)
            continue

        # Check whether this same change happened last time.
        previous_count, repeats = waiting.get(label, (None, 0))

        if previous_count == new_count:
            repeats += 1
        else:
            repeats = 1

        # Only accept the change once it's repeated enough.
        if repeats >= needed:
            if new_count > 0:
                updated[label] = new_count
            else:
                # A count of zero means it's out of sight.
                updated.pop(label, None)

            waiting.pop(label, None)
        else:
            waiting[label] = (new_count, repeats)

    # Tell the controller whether anything was confirmed.
    changed = updated != confirmed

    return updated, waiting or None, changed
