# aer/hypothesis/ha_gold.py

from typing import Dict, List

# Patterns representing "none-of-the-others" causal choices
NONE_PATTERNS = {
    "none",
    "none of the others",
    "none of these",
    "none of the above",
    "none of the other options",
    "no other option",
    "no option is correct",
    "no correct option",
}


def is_none_option(text: str) -> bool:
    """
    Returns True if the option text indicates a 'None-of-the-others'
    type of causal answer.
    """
    t = text.strip().lower()
    return any(pat in t for pat in NONE_PATTERNS)


def extract_cause_text(options: Dict[str, str], golden: List[str]) -> str:
    """
    Extract the normalized cause text used by BOTH H_a and H_w.
    """
    if not golden:
        return "the cause is not specified"

    gold_key = golden[0]

    if gold_key not in options:
        return "the correct cause is not available"

    gold_text = options[gold_key].strip()

    if is_none_option(gold_text):
        return "the correct cause is not among the provided options"

    return gold_text


def make_h_a(target_event: str, options: Dict[str, str], golden: List[str]) -> str:
    """
    Build abductive gold hypothesis Hₐ.
    """
    target_event = target_event.strip().rstrip(".")
    cause_text = extract_cause_text(options, golden)

    return f"{target_event}. This occurred because {cause_text}."


def make_h_w(target_event: str, options: Dict[str, str], golden: List[str]) -> str:
    """
    Build inverted abductive hypothesis H_w with identical structure to H_a.
    """
    target_event = target_event.strip().rstrip(".")
    cause_text = extract_cause_text(options, golden)

    return f"{target_event}. This occurred, but NOT because {cause_text}."
