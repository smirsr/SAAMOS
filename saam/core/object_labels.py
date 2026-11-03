"""Object labels the vision module can detect.

The vision owner (teammate 1) decides which labels exist. Put them, one per
line, in object_labels.txt next to this file (or point SAAM_LABELS_FILE at
another file). While the list is empty, every spoken object name is passed on
to vision unchanged.

Once the list is filled in, the controller will not call vision for an object
that is not on it. It tells the user "I can't look for that yet" instead of
claiming the object is not there.
"""

import logging
import os
from typing import Optional, Set

LOGGER = logging.getLogger(__name__)

ENV_LABELS_FILE = "SAAM_LABELS_FILE"
DEFAULT_LABELS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "object_labels.txt")


def load_supported_labels() -> Optional[Set[str]]:
    """Return the set of supported labels, or None if no list has been provided."""
    explicit = os.environ.get(ENV_LABELS_FILE)
    path = explicit or DEFAULT_LABELS_FILE
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as error:
        if explicit:
            LOGGER.warning("Cannot read %s=%r (%s); accepting any object name",
                           ENV_LABELS_FILE, explicit, error)
        return None
    labels = {line.strip().lower() for line in lines
              if line.strip() and not line.strip().startswith("#")}
    return labels or None


def is_supported_label(label: str) -> bool:
    """True if vision supports this label, or if no list has been provided yet."""
    labels = load_supported_labels()
    return labels is None or str(label).strip().lower() in labels
