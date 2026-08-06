"""Tests for narrashap.core.scorer."""

from __future__ import annotations

import pytest

from narrashap.core.extractor import ExplanationContext, FeatureContribution
from narrashap.core.scorer import extract_claims, score


def _context() -> ExplanationContext:
    return ExplanationContext(
        base_value=0.2,
        predicted_value=0.65,
        contributions=[
            FeatureContribution("smoking", 1, 0.30),
            FeatureContribution("age", 55, 0.20),
            FeatureContribution("bmi", 28.0, 0.10),
            FeatureContribution("exercise", 1, -0.05),
        ],
    )


class TestExtractClaims:
    def test_extracts_features_and_directions(self) -> None:
        text = (
            "Smoking increased the predicted risk. "
            "Regular exercise decreased the estimate."
        )
        claims = extract_claims(
            text,
            ["smoking", "age", "bmi", "exercise"],
        )

        by_feature = {c["feature"]: c["direction"] for c in claims}
        assert by_feature["smoking"] == "positive"
        assert by_feature["exercise"] == "negative"
        assert "age" not in by_feature

    def test_no_known_features_mentioned(self) -> None:
        text = "The model produced a moderate prediction."
        claims = extract_claims(text, ["smoking", "age"])
        assert claims == []


class TestScore:
    def test_high_scores_for_accurate_narrative(self) -> None:
        text = (
            "Smoking increased the predicted risk. "
            "Age also increased the estimate. "
            "BMI contributed to a higher score."
        )
        result = score(text, _context())

        assert result.completeness == pytest.approx(1.0)
        assert result.direction_accuracy == pytest.approx(1.0)
        assert result.hallucination_penalty == pytest.approx(0.0)
        assert result.overall == pytest.approx(1.0)

    def test_penalizes_wrong_direction(self) -> None:
        text = (
            "Smoking decreased the predicted risk. "
            "Age increased the estimate. "
            "BMI increased the estimate."
        )
        result = score(text, _context())

        assert result.direction_accuracy < 1.0

    def test_penalizes_hallucinated_feature(self) -> None:
        text = (
            "Smoking increased the risk. "
            "Age increased the risk. "
            "Income increased the risk."
        )
        known = ["smoking", "age", "bmi", "exercise", "income"]
        claims = extract_claims(text, known)
        assert any(c["feature"] == "income" for c in claims)

        context = _context()
        result = score(text, context)
        assert result.hallucination_penalty > 0.0

    def test_readability_simple_vs_complex(self) -> None:
        simple = "Risk went up. Smoking mattered."
        complex_text = (
            "The multifaceted interrelationship between chronic tobacco "
            "consumption and cardiovascular pathophysiology substantially "
            "exacerbated the algorithmic prognostication."
        )

        simple_grade = score(simple, _context()).readability
        complex_grade = score(complex_text, _context()).readability

        assert simple_grade is not None
        assert complex_grade is not None
        assert simple_grade < complex_grade
