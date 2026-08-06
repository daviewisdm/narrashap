"""Fidelity scoring for generated narratives against SHAP context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from narrashap.core.extractor import ExplanationContext

_POSITIVE_KEYWORDS = (
    "increased",
    "increase",
    "increases",
    "higher",
    "raised",
    "raised the",
    "elevated",
    "boosted",
    "positive",
    "upward",
    "greater",
)

_NEGATIVE_KEYWORDS = (
    "decreased",
    "decrease",
    "decreases",
    "lower",
    "reduced",
    "reduced the",
    "diminished",
    "negative",
    "downward",
    "less",
    "smaller",
)


@dataclass
class FidelityScore:
    """Component scores measuring narrative faithfulness to SHAP context."""

    completeness: float
    direction_accuracy: float
    hallucination_penalty: float
    readability: Optional[float] = None

    @property
    def overall(self) -> float:
        """Weighted combination of fidelity components (v1 default).

        Weights: 0.4 completeness + 0.4 direction_accuracy +
        0.2 * (1 - hallucination_penalty).

        These defaults are a starting point for v1 and should likely become
        configurable in a future release.
        """
        return (
            0.4 * self.completeness
            + 0.4 * self.direction_accuracy
            + 0.2 * (1.0 - self.hallucination_penalty)
        )


def _count_syllables(word: str) -> int:
    """Estimate syllable count using a simple vowel-group heuristic."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_kincaid_grade(text: str) -> float:
    """Compute Flesch-Kincaid grade level for *text*."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)

    if not sentences or not words:
        return 0.0

    syllable_count = sum(_count_syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllable_count / len(words)

    return 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59


def _sentence_containing(text: str, term: str) -> str:
    """Return the sentence that contains *term* (case-insensitive)."""
    lower_term = term.lower()
    for sentence in re.split(r"[.!?]+", text):
        if lower_term in sentence.lower():
            return sentence
    return ""


def _detect_direction_near_feature(narrative_text: str, feature: str) -> str:
    """Infer claim direction from keywords in the same sentence as *feature*."""
    sentence = _sentence_containing(narrative_text, feature)
    if not sentence:
        return "unclear"

    lower_sentence = sentence.lower()
    has_positive = any(kw in lower_sentence for kw in _POSITIVE_KEYWORDS)
    has_negative = any(kw in lower_sentence for kw in _NEGATIVE_KEYWORDS)

    if has_positive and not has_negative:
        return "positive"
    if has_negative and not has_positive:
        return "negative"
    return "unclear"


_DIRECTION_STOP_WORDS = frozenset(
    {"the", "a", "an", "this", "that", "also", "which", "and", "or", "regular"}
)


def _features_mentioned_with_direction(narrative_text: str) -> list[str]:
    """Find feature-like tokens immediately before direction language."""
    pattern = re.compile(
        r"\b([A-Za-z][A-Za-z0-9_]*)\s+"
        r"(increased|decreased|increase|decrease|raised|lowered|reduced|elevated|boosted)\b",
        re.IGNORECASE,
    )
    return [
        match.group(1)
        for match in pattern.finditer(narrative_text)
        if match.group(1).lower() not in _DIRECTION_STOP_WORDS
    ]


def _is_known_feature(name: str, context_names: list[str]) -> bool:
    """Return True if *name* matches a feature in *context_names*."""
    lower = name.lower()
    for known in context_names:
        known_lower = known.lower()
        if lower == known_lower or known_lower in lower or lower in known_lower:
            return True
    return False


def extract_claims(narrative_text: str, known_features: list[str]) -> list[dict]:
    """Extract feature-direction claims from *narrative_text* via keyword matching.

    Parameters
    ----------
    narrative_text:
        Generated narrative to analyze.
    known_features:
        Feature names to search for in the text.

    Returns
    -------
    list[dict]
        Each dict has keys ``"feature"`` and ``"direction"``
        (``"positive"``, ``"negative"``, or ``"unclear"``).
    """
    claims: list[dict] = []
    lower_text = narrative_text.lower()

    for feature in known_features:
        if feature.lower() not in lower_text:
            continue
        direction = _detect_direction_near_feature(narrative_text, feature)
        claims.append({"feature": feature, "direction": direction})

    return claims


def score(narrative_text: str, context: ExplanationContext) -> FidelityScore:
    """Score *narrative_text* against *context* SHAP attributions.

    Parameters
    ----------
    narrative_text:
        Generated narrative to evaluate.
    context:
        Ground-truth SHAP explanation context.

    Returns
    -------
    FidelityScore
        Component fidelity metrics and derived overall score.
    """
    context_features = [c.name for c in context.contributions]
    claims = extract_claims(narrative_text, context_features)

    top_three = context.contributions[:3]
    top_names = {c.name for c in top_three}

    mentioned_top = sum(
        1 for c in claims if c["feature"] in top_names
    )
    completeness = mentioned_top / len(top_three) if top_three else 1.0

    shap_by_name = {c.name: c.shap_value for c in context.contributions}

    direction_matches = 0
    direction_total = 0
    for claim in claims:
        feature = claim["feature"]
        if claim["direction"] == "unclear":
            continue
        if feature not in shap_by_name:
            continue
        direction_total += 1
        true_positive = shap_by_name[feature] >= 0
        claimed_positive = claim["direction"] == "positive"
        if true_positive == claimed_positive:
            direction_matches += 1

    direction_accuracy = (
        direction_matches / direction_total if direction_total > 0 else 1.0
    )

    direction_mentions = _features_mentioned_with_direction(narrative_text)
    unknown_mentions = [
        m for m in direction_mentions if not _is_known_feature(m, context_features)
    ]
    total_for_hallucination = len(claims) + len(unknown_mentions)
    hallucinated = sum(
        1 for claim in claims if claim["feature"] not in shap_by_name
    ) + len(unknown_mentions)
    hallucination_penalty = (
        hallucinated / total_for_hallucination if total_for_hallucination > 0 else 0.0
    )

    readability = _flesch_kincaid_grade(narrative_text)

    return FidelityScore(
        completeness=completeness,
        direction_accuracy=direction_accuracy,
        hallucination_penalty=hallucination_penalty,
        readability=readability,
    )
