"""Text-to-intent parsing for the SAAM controller.

parse_intent() tries the small language model first (see slm_engine.py). If the
model is unavailable, too slow, or returns something invalid, it falls back to
simple keyword/regex rules, so the rest of the system keeps working.

Callers always get the same format, whichever method produced it:

    {"intent": "FIND_OBJECT", "target": "bottle"}
"""

import logging
import re
from typing import Dict, Optional

from . import slm_engine

LOGGER = logging.getLogger(__name__)

FIND_OBJECT = "FIND_OBJECT"
DESCRIBE_SCENE = "DESCRIBE_SCENE"
STOP = "STOP"
UNKNOWN = "UNKNOWN"

SUPPORTED_INTENTS = (FIND_OBJECT, DESCRIBE_SCENE, STOP)

# Example object vocabulary. Agree on the real list of names with the vision
# owner (teammate 1) and edit these tables to match.
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

STOP_WORDS = {"stop", "cancel", "quiet", "enough", "halt", "nevermind"}
STOP_PHRASES = ("never mind", "be quiet", "shut up")

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

FIND_PATTERNS = [
    r"\bwhere(?:'s| is| are| did i put| did i leave| can i find)\s+(?P<t>.+)$",
    r"\b(?:find|locate|spot|look for|looking for|search for)\s+(?P<t>.+)$",
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


def _clean_target(phrase: str) -> Optional[str]:
    phrase = phrase.strip()
    previous = None
    while phrase != previous:
        previous = phrase
        phrase = _LEADING.sub("", phrase)
        phrase = _TRAILING.sub("", phrase)
    return normalize_target(phrase)


def _is_stop(normalized: str) -> bool:
    tokens = normalized.split()
    if not tokens or len(tokens) > 4 or "sign" in tokens:
        return False
    if any(token in STOP_WORDS for token in tokens):
        return True
    return any(phrase in normalized for phrase in STOP_PHRASES)


def rule_based_parse(text: str) -> Dict:
    """Keyword/regex parser. Used when the SLM is unavailable or fails."""
    normalized = _normalize(text or "")
    if not normalized:
        return _result(UNKNOWN)
    if _is_stop(normalized):
        return _result(STOP)
    for pattern in SCENE_PATTERNS:
        if re.search(pattern, normalized):
            return _result(DESCRIBE_SCENE)
    for pattern in FIND_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            target = _clean_target(match.group("t"))
            if target:
                return _result(FIND_OBJECT, target)
    return _result(UNKNOWN)


def _validate(candidate) -> Optional[Dict]:
    """Check SLM output. Returns a clean intent dict, or None if unusable."""
    if not isinstance(candidate, dict):
        return None
    intent = str(candidate.get("intent", "")).strip().upper()
    if intent in SUPPORTED_INTENTS or intent == UNKNOWN:
        pass
    else:
        return None
    if intent == FIND_OBJECT:
        target = candidate.get("target")
        if not isinstance(target, str):
            return None
        target = normalize_target(_normalize(target))
        if not target:
            return None
        return _result(FIND_OBJECT, target)
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
