"""Causal-language guardrails for narrative generation."""

from __future__ import annotations

import re

BANNED_PHRASES: list[str] = [
    "caused",
    "causes",
    "will cause",
    "leads to",
    "led to",
    "results in",
    "resulted in",
    "proves",
    "proven",
    "definitely",
    "guarantees",
    "guaranteed",
    "because of this",
    "due to this the patient will",
    "will develop",
    "will get",
]

REQUIRED_HEDGE_EXAMPLES: list[str] = [
    "associated with",
    "contributed to",
    "may be linked to",
    "appears to have influenced",
    "is correlated with",
    "was among the factors that",
]

# Negation words checked anywhere earlier in the same sentence as a banned
# phrase — not just a fixed word-count lookback, since real hedge phrasing
# like "This does not mean that smoking causes fibroids" puts several
# words between "not" and the banned word.
NEGATION_WORDS = {
    "not", "never", "no", "n't", "isn't", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "cannot", "can't", "couldn't", "without",
}

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]")
_WORD_RE = re.compile(r"[a-z']+")


def _sentence_containing(lower_text: str, match_start: int) -> str:
    """Return the sentence fragment from its start up to match_start."""
    boundaries = [m.end() for m in _SENTENCE_SPLIT_RE.finditer(lower_text, 0, match_start)]
    sentence_start = boundaries[-1] if boundaries else 0
    return lower_text[sentence_start:match_start]


def _is_negated(lower_text: str, match_start: int) -> bool:
    """Return True if a negation word appears earlier in the same sentence."""
    fragment = _sentence_containing(lower_text, match_start)
    words = _WORD_RE.findall(fragment)
    return any(word in NEGATION_WORDS for word in words)


def check_narrative(text: str) -> list[str]:
    """Return banned phrases found in *text* (case-insensitive substring match).

    A banned phrase that has a negation word earlier in the same sentence
    (e.g. "not proven to cause", "does not mean ... causes") is treated as
    a safe hedge, not a violation — this avoids false positives on exactly
    the kind of disclaimer language REQUIRED_HEDGE_EXAMPLES asks the model
    to use.

    Note: this is sentence-scoped, not a fixed-word lookback, so it handles
    longer negated clauses correctly. One known tradeoff: a compound
    sentence with an unrelated negation earlier in the same sentence (e.g.
    "It does not lower X; however, it causes Y.") could be incorrectly
    cleared. Judged an acceptable rare edge case versus the false-positive
    rate of a short fixed-word lookback.

    Parameters
    ----------
    text:
        Generated narrative text to inspect.

    Returns
    -------
    list[str]
        Banned phrases detected in *text* (unnegated occurrences only),
        or an empty list if clean.
    """
    lower_text = text.lower()
    found: list[str] = []
    for phrase in BANNED_PHRASES:
        phrase_lower = phrase.lower()
        start = 0
        while True:
            idx = lower_text.find(phrase_lower, start)
            if idx == -1:
                break
            if not _is_negated(lower_text, idx):
                if phrase not in found:
                    found.append(phrase)
                break
            start = idx + len(phrase_lower)
    return found