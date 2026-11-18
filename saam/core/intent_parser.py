"""Text-to-intent parsing for the SAAM controller.

parse_intent() tries the small language model first (see slm_engine.py). If the
model is unavailable, too slow, or returns something invalid, it falls back to
simple keyword/regex rules, so the rest of the system keeps working.

Callers always get the same format, whichever method produced it:

    {"intent": "FIND_OBJECT", "target": "bottle"}

One-shot vs continuous commands
-------------------------------
    "Where is my bottle?"    -> FIND_OBJECT  (answer once)
    "Find my bottle"         -> START_FIND   (keep looking until stopped)
    "Stop searching"         -> STOP_FIND
    "Scan my surroundings"   -> START_SCAN   (keep watching for changes)
    "Stop scanning"          -> STOP_SCAN
    "How many bottles ...?"  -> COUNT_OBJECTS (target is singular: "bottle")
    "Read this text"         -> READ_TEXT
    "Repeat that"            -> REPEAT
    "Stop"                   -> STOP (end whatever is happening)
"""

import logging
import re
from typing import Dict, Optional

from . import slm_engine

LOGGER = logging.getLogger(__name__)

# --- Original intents (unchanged) ---
FIND_OBJECT = "FIND_OBJECT"
DESCRIBE_SCENE = "DESCRIBE_SCENE"
STOP = "STOP"
UNKNOWN = "UNKNOWN"

# --- New intents for operating modes ---
START_FIND = "START_FIND"
STOP_FIND = "STOP_FIND"
START_SCAN = "START_SCAN"
STOP_SCAN = "STOP_SCAN"
COUNT_OBJECTS = "COUNT_OBJECTS"
READ_TEXT = "READ_TEXT"
REPEAT = "REPEAT"

SUPPORTED_INTENTS = (
    FIND_OBJECT, START_FIND, STOP_FIND, START_SCAN, STOP_SCAN,
    COUNT_OBJECTS, DESCRIBE_SCENE, READ_TEXT, REPEAT, STOP,
)
# Intents that must carry an object name in "target".
INTENTS_WITH_TARGET = (FIND_OBJECT, START_FIND, COUNT_OBJECTS)

# Example object vocabulary. Agree on the real list of names with the vision
# owner and edit these tables to match (see also object_labels.txt).
MULTIWORD_LABELS = {
    "cell phone", "stop sign", "wine glass", "potted plant", "dining table",
    "teddy bear", "traffic light", "fire hydrant", "parking meter",
    "sports ball", "baseball bat", "tennis racket", "hair drier",
}
ALIASES = {
    "water bottle": "bottle",
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "cellphone": "cell phone",
    "television": "tv",
    "sofa": "couch",
    "mug": "cup",
    "coffee cup": "cup",
}

# Plural -> singular for counting questions ("How many bottles ...?").
IRREGULAR_PLURALS = {
    "people": "person", "men": "man", "women": "woman", "mice": "mouse",
    "knives": "knife", "leaves": "leaf", "shelves": "shelf", "tvs": "tv",
}
NO_SINGULAR = {"scissors", "jeans", "pants", "shorts"}
# "How many objects ...?" has no single label, so it is answered as a scene.
GENERIC_WORDS = {"object", "thing", "item", "stuff"}

STOP_WORDS = {"stop", "cancel", "quiet", "enough", "halt", "nevermind"}
STOP_PHRASES = ("never mind", "be quiet", "shut up")

STOP_FIND_PATTERNS = [
    r"\b(?:stop|cancel|end|quit)\s+(?:the\s+|my\s+)?(?:searching|search|looking|finding|find)\b",
]
STOP_SCAN_PATTERNS = [
    r"\b(?:stop|cancel|end|quit)\s+(?:the\s+|my\s+)?(?:scanning|scan|monitoring|watching)\b",
]
START_SCAN_PATTERNS = [
    r"\bscan(?:ning)?\b",
    r"\bmonitor (?:my |the )?(?:surroundings|room|area|environment)\b",
]
READ_TEXT_PATTERNS = [
    r"\bread\b.*\b(?:this|that|it|text|sign|label|words|page|aloud)\b",
    r"\bwhat does (?:this|that|it) say\b",
]
REPEAT_PATTERNS = [
    r"^(?:please )?repeat(?: that| it| this)?(?: please)?$",
    r"\bsay (?:that|it) again\b",
    r"\bwhat did you (?:just )?say\b",
    r"^(?:pardon|come again)$",
]
COUNT_PATTERN = (
    r"\bhow many\s+(?P<t>[a-z' \-]+?)"
    r"(?:\s+(?:do|does|can|could|are|is|did|in|on|at|here|there|around|near|nearby|have|am|will)\b.*)?$"
)
# Words that mean the "object" captured by COUNT_PATTERN is really a sentence fragment.
COUNT_FILLER_WORDS = {"do", "does", "you", "see", "i", "are", "is", "there", "can", "many"}

SCENE_PATTERNS = [
    r"\bwhat\b.*\b(?:objects?|things?|items?)\b.*\b(?:in front|around|near|nearby|here|see|ahead)\b",
    r"\bwhat (?:do|can) (?:you|i) see\b",
    r"\b(?:can|do) you see (?:anything|something)\b",
    r"\bwhat(?:'s| is)? (?:in front|around|near|nearby|ahead)\b",
    r"\btell me what(?:'s| is)? (?:around|in front|nearby|here|there|ahead)\b",
    r"\bwhat(?:'s| is| am i) (?:looking at|facing)\b",
    r"\bwhat(?:'s| is) (?:in )?(?:the |this )?(?:room|scene|view)\b",
    r"\bis there (?:anything|something|any objects?)\b",
    r"\bdescribe\b",
]

# "Where is my bottle?" -> answer once.
WHERE_PATTERN = r"\bwhere(?:'s| is| are| did i put| did i leave| can i find)\s+(?P<t>.+)$"
# "Find my bottle" -> keep looking until the user stops it.
CONTINUOUS_FIND_PATTERN = r"\b(?:find|locate|spot|look for|looking for|search for)\s+(?P<t>.+)$"
# Other questions about one object -> answer once.
ONE_SHOT_FIND_PATTERNS = [
    r"\b(?:can|do) you see\s+(?P<t>.+)$",
    r"\bis there\s+(?P<t>.+)$",
    r"\b(?:have you seen|i lost|i misplaced|i can't find|i cannot find)\s+(?P<t>.+)$",
]

_LEADING = re.compile(r"^(?:the|my|a|an|some|any|your|that|this|those|these|our|me)\s+")
_TRAILING = re.compile(
    r"\s+(?:please|now|right now|for me|nearby|near me|around me|around here|"
    r"around|here|there|in front of me|in front|close by|at the moment)$"
)


def _result(intent: str, target: Optional[str] = None) -> Dict:
    return {"intent": intent, "target": target}


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation (keeping apostrophes) and extra spaces."""
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9'\- ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_target(phrase: str) -> Optional[str]:
    """Turn a spoken object phrase into one object name, or None.

    "water bottle" -> "bottle", "red mug" -> "cup". Phrases longer than four
    words are treated as not understood.
    """
    phrase = phrase.strip().lower()
    words = phrase.split()
    if not words or len(words) > 4 or not re.fullmatch(r"[a-z' \-]+", phrase):
        return None
    if phrase in MULTIWORD_LABELS:
        return phrase
    if phrase in ALIASES:
        return ALIASES[phrase]
    if len(words) >= 2:
        tail = " ".join(words[-2:])
        if tail in MULTIWORD_LABELS:
            return tail
        if tail in ALIASES:
            return ALIASES[tail]
    last = words[-1]
    return ALIASES.get(last, last)


def singularize(word: str) -> str:
    """Very small plural -> singular helper ("bottles" -> "bottle")."""
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if word in NO_SINGULAR:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if re.search(r"(?:ses|xes|zes|ches|shes)$", word):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _clean_target(phrase: str) -> Optional[str]:
    phrase = phrase.strip()
    previous = None
    while phrase != previous:
        previous = phrase
        phrase = _LEADING.sub("", phrase)
        phrase = _TRAILING.sub("", phrase)
    return normalize_target(phrase)


def _clean_count_target(phrase: str) -> Optional[str]:
    """Clean the object in "How many <objects> do you see?" (plural -> singular)."""
    phrase = _LEADING.sub("", phrase.strip())
    words = phrase.split()
    if not words or words[0] in COUNT_FILLER_WORDS:
        return None
    words[-1] = singularize(words[-1])
    return normalize_target(" ".join(words))


def _is_stop(normalized: str) -> bool:
    tokens = normalized.split()
    if not tokens or len(tokens) > 4 or "sign" in tokens:
        return False
    if any(token in STOP_WORDS for token in tokens):
        return True
    return any(phrase in normalized for phrase in STOP_PHRASES)


def _matches(patterns, normalized: str) -> bool:
    return any(re.search(pattern, normalized) for pattern in patterns)


def _find_target(pattern: str, normalized: str) -> Optional[str]:
    match = re.search(pattern, normalized)
    return _clean_target(match.group("t")) if match else None


def rule_based_parse(text: str) -> Dict:
    """Keyword/regex parser. Used when the SLM is unavailable or fails.

    Order matters: more specific commands ("stop searching") are checked
    before more general ones ("stop"), and continuous commands ("find ...")
    before one-shot questions.
    """
    normalized = _normalize(text or "")
    if not normalized:
        return _result(UNKNOWN)

    if _matches(REPEAT_PATTERNS, normalized):
        return _result(REPEAT)
    if _matches(STOP_FIND_PATTERNS, normalized):
        return _result(STOP_FIND)
    if _matches(STOP_SCAN_PATTERNS, normalized):
        return _result(STOP_SCAN)
    if _is_stop(normalized):
        return _result(STOP)
    if _matches(READ_TEXT_PATTERNS, normalized):
        return _result(READ_TEXT)
    if _matches(START_SCAN_PATTERNS, normalized):
        return _result(START_SCAN)

    count = re.search(COUNT_PATTERN, normalized)
    if count:
        raw = _LEADING.sub("", count.group("t").strip())
        if raw and singularize(raw.split()[-1]) in GENERIC_WORDS:
            return _result(DESCRIBE_SCENE)
        target = _clean_count_target(count.group("t"))
        if target:
            return _result(COUNT_OBJECTS, target)

    if _matches(SCENE_PATTERNS, normalized):
        return _result(DESCRIBE_SCENE)

    target = _find_target(WHERE_PATTERN, normalized)
    if target:
        return _result(FIND_OBJECT, target)
    target = _find_target(CONTINUOUS_FIND_PATTERN, normalized)
    if target:
        return _result(START_FIND, target)
    for pattern in ONE_SHOT_FIND_PATTERNS:
        target = _find_target(pattern, normalized)
        if target:
            return _result(FIND_OBJECT, target)
    return _result(UNKNOWN)


def _validate(candidate) -> Optional[Dict]:
    """Check SLM output. Returns a clean intent dict, or None if unusable."""
    if not isinstance(candidate, dict):
        return None
    intent = str(candidate.get("intent", "")).strip().upper()
    if intent not in SUPPORTED_INTENTS and intent != UNKNOWN:
        return None
    if intent in INTENTS_WITH_TARGET:
        target = candidate.get("target")
        if not isinstance(target, str):
            return None
        target = normalize_target(_normalize(target))
        if not target:
            return None
        return _result(intent, target)
    return _result(intent)


def parse_intent(text: str, use_slm: bool = True) -> Dict:
    """Turn recognized text into {"intent": ..., "target": ...}.

    Tries the SLM first, then the rule-based parser. Never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return _result(UNKNOWN)

    if use_slm:
        try:
            candidate = slm_engine.slm_parse(text)
        except Exception:  # the SLM must never crash the pipeline
            LOGGER.exception("SLM failed; using rule-based fallback")
            candidate = None
        result = _validate(candidate)
        if result is not None and result["intent"] != UNKNOWN:
            LOGGER.debug("intent from SLM: %s", result)
            return result
        if candidate is not None:
            LOGGER.debug("SLM output unusable (%r); using fallback", candidate)

    result = rule_based_parse(text)
    LOGGER.debug("intent from rules: %s", result)
    return result
