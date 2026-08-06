"""Causal-language guardrails for narrative generation."""

from __future__ import annotations

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


def check_narrative(text: str) -> list[str]:
    """Return banned phrases found in *text* (case-insensitive substring match).

    Parameters
    ----------
    text:
        Generated narrative text to inspect.

    Returns
    -------
    list[str]
        Banned phrases detected in *text*, or an empty list if clean.
    """
    lower_text = text.lower()
    found: list[str] = []
    for phrase in BANNED_PHRASES:
        if phrase.lower() in lower_text:
            found.append(phrase)
    return found
