"""Tests for narrashap.core.narrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from narrashap.core.extractor import ExplanationContext, FeatureContribution
from narrashap.core.narrator import BaseNarrator, Narrative
from narrashap.core.scorer import FidelityScore
from narrashap.templates.causal_language import BANNED_PHRASES


def _sample_context() -> ExplanationContext:
    return ExplanationContext(
        base_value=0.3,
        predicted_value=0.7,
        contributions=[
            FeatureContribution("smoking", 1, 0.25, percentile=85.0),
            FeatureContribution("age", 62, 0.15, percentile=70.0),
            FeatureContribution("bmi", 31.0, 0.10, percentile=60.0),
            FeatureContribution("exercise", 0, -0.08, percentile=20.0),
        ],
    )


class TestBuildPrompt:
    def test_includes_top_contributions(self) -> None:
        narrator = BaseNarrator(audience="clinician")
        prompt = narrator.build_prompt(_sample_context())

        assert "smoking" in prompt
        assert "age" in prompt
        assert "bmi" in prompt
        assert "base_value" in prompt.lower() or "Baseline" in prompt
        assert "0.3000" in prompt or "0.3" in prompt

    def test_unknown_audience_raises(self) -> None:
        narrator = BaseNarrator(audience="unknown_audience")
        with pytest.raises(ValueError, match="Unknown audience"):
            narrator.build_prompt(_sample_context())

    def test_includes_banned_phrase_instruction(self) -> None:
        narrator = BaseNarrator(audience="clinician")
        prompt = narrator.build_prompt(_sample_context())

        assert "banned" in prompt.lower()
        assert BANNED_PHRASES[0] in prompt


class TestExplain:
    def test_raises_without_llm_client(self) -> None:
        narrator = BaseNarrator(audience="clinician", llm_client=None)
        with pytest.raises(ValueError, match="LLM client is required"):
            narrator.explain(_sample_context())

    @patch("narrashap.core.narrator.score")
    def test_explain_with_clean_mock_llm(self, mock_score: MagicMock) -> None:
        mock_score.return_value = FidelityScore(
            completeness=1.0,
            direction_accuracy=1.0,
            hallucination_penalty=0.0,
            readability=10.0,
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "Smoking history contributed to a higher estimated risk, "
            "while exercise was associated with a lower estimate."
        )

        narrator = BaseNarrator(audience="clinician", llm_client=mock_client)
        result = narrator.explain(_sample_context())

        assert isinstance(result, Narrative)
        assert result.audience == "clinician"
        assert result.fidelity_score == pytest.approx(1.0)
        assert result.fairness_flags is None
        mock_client.generate.assert_called_once()

    @patch("narrashap.core.narrator.score")
    def test_explain_raises_on_persistent_banned_phrases(
        self,
        mock_score: MagicMock,
    ) -> None:
        mock_score.return_value = FidelityScore(
            completeness=0.5,
            direction_accuracy=0.5,
            hallucination_penalty=0.0,
        )

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "Smoking caused the high risk prediction."
        )

        narrator = BaseNarrator(audience="clinician", llm_client=mock_client)
        with pytest.raises(RuntimeError, match="banned phrases"):
            narrator.explain(_sample_context())

        assert mock_client.generate.call_count == 2
