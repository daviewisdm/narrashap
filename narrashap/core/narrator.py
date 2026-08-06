"""Base narrator for turning ExplanationContext into Narrative objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from narrashap.core.extractor import ExplanationContext
from narrashap.core.llm_client import LLMClient
from narrashap.core.scorer import score
from narrashap.templates.audience import AUDIENCE_PROFILES
from narrashap.templates.causal_language import (
    BANNED_PHRASES,
    REQUIRED_HEDGE_EXAMPLES,
    check_narrative,
)


@dataclass
class Narrative:
    """A generated explanation narrative with optional quality metadata."""

    text: str
    audience: str
    fidelity_score: Optional[float] = None
    fairness_flags: Optional[list[str]] = None


class BaseNarrator:
    """Build prompts and generate fidelity-checked narratives from SHAP context."""

    terminology_map: dict[str, str] = {}
    causal_language_policy: dict[str, list[str]] = {}
    tone_profile: str = "neutral"

    def __init__(
        self,
        audience: str = "clinician",
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.audience = audience
        self.llm_client = llm_client

    def _all_banned_phrases(self) -> list[str]:
        """Merge shared and domain-specific banned phrases."""
        domain_banned = self.causal_language_policy.get("banned", [])
        return list(BANNED_PHRASES) + list(domain_banned)

    def _all_hedge_examples(self) -> list[str]:
        """Merge shared and domain-specific hedge examples."""
        domain_hedges = self.causal_language_policy.get("required_hedges", [])
        return list(REQUIRED_HEDGE_EXAMPLES) + list(domain_hedges)

    def build_prompt(self, context: ExplanationContext) -> str:
        """Construct an LLM prompt from *context* and narrator configuration.

        Parameters
        ----------
        context:
            Structured SHAP explanation for a single instance.

        Returns
        -------
        str
            Prompt string for the LLM client.

        Raises
        ------
        ValueError
            If ``self.audience`` is not a key in :data:`AUDIENCE_PROFILES`.
        """
        if self.audience not in AUDIENCE_PROFILES:
            valid = ", ".join(sorted(AUDIENCE_PROFILES))
            raise ValueError(
                f"Unknown audience '{self.audience}'. Valid audiences: {valid}"
            )

        target_grade = AUDIENCE_PROFILES[self.audience]
        top_contributions = context.contributions[:5]

        contribution_lines: list[str] = []
        for contrib in top_contributions:
            direction = "positive" if contrib.shap_value >= 0 else "negative"
            percentile_str = (
                f", percentile={contrib.percentile:.1f}"
                if contrib.percentile is not None
                else ""
            )
            contribution_lines.append(
                f"- {contrib.name}: value={contrib.value}, "
                f"shap={contrib.shap_value:+.4f}, direction={direction}"
                f"{percentile_str}"
            )

        terminology_lines = [
            f"  '{generic}' -> '{domain}'"
            for generic, domain in self.terminology_map.items()
        ]

        banned = self._all_banned_phrases()
        hedges = self._all_hedge_examples()

        prompt_parts = [
            "You are generating a faithful narrative explanation of a machine "
            "learning model prediction based on SHAP feature attributions.",
            "",
            f"Tone profile: {self.tone_profile}",
            f"Target audience: {self.audience}",
            f"Target reading level: Flesch-Kincaid grade {target_grade}",
            "",
            "Use the following terminology substitutions where generic ML "
            "concepts appear:",
            *terminology_lines,
            "",
            f"Baseline prediction (base_value): {context.base_value:.4f}",
            f"Final predicted value: {context.predicted_value:.4f}",
            "",
            "Top feature contributions (by absolute SHAP value):",
            *contribution_lines,
            "",
            "IMPORTANT — causal language restrictions:",
            "Do NOT use any of the following banned phrases:",
            ", ".join(f'"{p}"' for p in banned),
            "",
            "Instead, use hedging language such as:",
            ", ".join(f'"{h}"' for h in hedges),
            "",
            "Write 2-4 sentences explaining how the model reached this "
            "prediction. Attribute changes to features using association "
            "language only — never imply causation.",
        ]
        return "\n".join(prompt_parts)

    def explain(self, context: ExplanationContext) -> Narrative:
        """Generate a narrative for *context* via the configured LLM client.

        Parameters
        ----------
        context:
            Structured SHAP explanation for a single instance.

        Returns
        -------
        Narrative
            Generated text with fidelity score attached.

        Raises
        ------
        ValueError
            If no LLM client was supplied at construction time.
        RuntimeError
            If the LLM output still contains banned phrases after one retry.
        """
        if self.llm_client is None:
            raise ValueError(
                "An LLM client is required. Pass llm_client= to the narrator "
                "constructor (e.g. AnthropicClient or a mock for testing)."
            )

        prompt = self.build_prompt(context)
        text = self.llm_client.generate(prompt)

        banned_found = check_narrative(text)
        domain_banned = self.causal_language_policy.get("banned", [])
        lower_text = text.lower()
        for phrase in domain_banned:
            if phrase.lower() in lower_text and phrase not in banned_found:
                banned_found.append(phrase)

        if banned_found:
            retry_prompt = (
                prompt
                + "\n\nRETRY INSTRUCTION: Your previous response contained "
                "banned causal phrases: "
                + ", ".join(f'"{p}"' for p in banned_found)
                + ". Rewrite without using any of those phrases."
            )
            text = self.llm_client.generate(retry_prompt)

            banned_found = check_narrative(text)
            for phrase in domain_banned:
                if phrase.lower() in text.lower() and phrase not in banned_found:
                    banned_found.append(phrase)

            if banned_found:
                raise RuntimeError(
                    "Generated narrative still contains banned phrases after "
                    f"retry: {banned_found}"
                )

        fidelity = score(text, context)

        return Narrative(
            text=text,
            audience=self.audience,
            fidelity_score=fidelity.overall,
            fairness_flags=None,
        )
