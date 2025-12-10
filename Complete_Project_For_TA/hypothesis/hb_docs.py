# aer/hypothesis/hb_docs.py

import re
from typing import List
from collections import Counter

# =========================================================
# 0. PARAMETERS
# =========================================================

CAUSAL_MARKERS = [
    "because", "after", "when", "as a result", "therefore",
    "thus", "led to", "resulted", "due to", "hence", "consequently"
]

# Irrelevant violent/chaotic events that often pollute news topics
NOISE_TERMS = [
    "killed", "dead", "died", "shooting", "attack",
    "gunman", "bomb", "explosion", "suicide", "fatal",
    "wounded", "injured", "stabbed"
]


# =========================================================
# 1. EDU EXTRACTION
# =========================================================

def extract_edus(text: str) -> List[str]:
    """
    Extract simple EDUs by:
      - whitespace normalization
      - injecting boundaries around causal markers
      - splitting on [.;]
    """
    t = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

    # Force boundaries around causal markers
    for m in CAUSAL_MARKERS:
        t = t.replace(f" {m} ", f". {m} ")

    parts = re.split(r"[.;]", t)

    edus = [p.strip() for p in parts if len(p.strip()) > 4]
    return edus


# =========================================================
# 2. KEYWORD EXTRACTION
# =========================================================

def build_keywords(topic_text: str, target_event: str) -> List[str]:
    """
    Extract stable, semantically relevant keywords from:
      - topic text
      - target event
    """
    base = f"{topic_text} {target_event}".lower()
    base = re.sub(r"[^a-zA-Z0-9\s]", " ", base)

    words = [
        w for w in base.split()
        if len(w) > 3 and w not in {
            "this", "that", "from", "with", "have", "were",
            "been", "will", "said", "they", "their"
        }
    ]

    return list(set(words))


# =========================================================
# 3. STRICT EDU SCORING
# =========================================================

def score_edu_strict(edu: str, keywords: List[str]) -> float:
    edu_l = edu.lower()
    score = 0.0

    # Keyword hits (strong signal)
    kw_hits = sum(1 for kw in keywords if kw in edu_l)
    score += kw_hits * 4.5

    # Light causal marker bonus
    if any(m in edu_l for m in CAUSAL_MARKERS):
        score += 0.75

    # Noise penalty (only if weak topicality)
    if any(nt in edu_l for nt in NOISE_TERMS):
        if kw_hits < 2:
            score -= 4.0

    # Small length bonus (caps at 140 chars)
    score += min(len(edu), 140) / 100.0

    return score


# =========================================================
# 4. SELECT TOP EDU (STRICT ABSTRACTION)
# =========================================================

def select_top_edus(
    target_event: str,
    topic_text: str,
    snippets: List[str],
    max_edus: int = 1
) -> List[str]:

    keywords = build_keywords(topic_text, target_event)

    # Collect all EDUs
    all_edus = []
    for s in snippets:
        all_edus.extend(extract_edus(s))

    if not all_edus:
        return []

    # Prefer EDUs with >= 2 keyword matches
    strict = []
    for edu in all_edus:
        hits = sum(1 for kw in keywords if kw in edu.lower())
        if hits >= 2:
            strict.append(edu)

    # Fallback: >= 1 keyword hit
    if not strict:
        strict = [
            e for e in all_edus
            if sum(1 for kw in keywords if kw in e.lower()) >= 1
        ]

    if not strict:
        return []

    # Score + Sort
    scored = [(score_edu_strict(e, keywords), e) for e in strict]
    scored.sort(reverse=True, key=lambda x: x[0])

    # Return top K
    return [edu for _, edu in scored[:max_edus]]


# =========================================================
# 5. CONSTRUCT H_b
# =========================================================

def make_h_b(
    target_event: str,
    topic_text: str,
    snippets: List[str],
    max_edus: int = 1
) -> str:
    """
    Build H_b with:
      - EXACTLY ONE EDU by default.
      - No hallucination.
      - Symmetric nucleus with H_a + H_w.
    """

    event = target_event.strip().rstrip(".")

    top_edus = select_top_edus(
        target_event=target_event,
        topic_text=topic_text,
        snippets=snippets or [],
        max_edus=max_edus
    )

    if not top_edus:
        return f"{event}."

    edu = top_edus[0].strip().rstrip(".")

    return (
        f"{event}. "
        f"This is supported by reports that {edu}."
    )
