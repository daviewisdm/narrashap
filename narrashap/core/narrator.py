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

# Jargon that technically satisfies the causal-language rules but still
# reads as a statistics report, not a plain-language explanation. Kept
# separate from BANNED_PHRASES since it's about tone/accessibility, not
# causal-claim safety.
JARGON_TERMS = [
    "SHAP",
    "attribution weight",
    "attribution",
    "log-odds",
    "baseline prediction",
    "positive attribution",
    "negative attribution",
    "risk attributions",
    "feature",
]


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

    def build_prompt(
        self,
        context: ExplanationContext,
        risk_percentage: Optional[float] = None,
        risk_level: Optional[str] = None,
    ) -> str:
        """Construct an LLM prompt from *context* and narrator configuration.

        Parameters
        ----------
        context:
            Structured SHAP explanation for a single instance.
        risk_percentage:
            Optional real-world risk percentage (e.g. 65.6, meaning 65.6%).
            When provided, this is shown to the LLM instead of the raw
            base_value/predicted_value, which are in log-odds space and
            not meaningful to a non-technical reader — and tend to get
            echoed back verbatim if given directly.
        risk_level:
            Optional label like "MODERATE RISK" to include for context.

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

        # Deliberately omit raw SHAP magnitudes and percentiles from what
        # the LLM sees — only direction and relative rank. This keeps the
        # model from repeating numbers back as jargon (e.g. "attribution
        # weight of +1.58"). The real numbers are still used separately
        # by the fidelity scorer against the actual context, not the prompt.
        contribution_lines: list[str] = []
        for rank, contrib in enumerate(top_contributions, start=1):
            direction = "increases the risk" if contrib.shap_value >= 0 else "decreases the risk"
            contribution_lines.append(f"- Rank {rank}: {contrib.name} — {direction}")

        terminology_lines = [
            f"  '{generic}' -> '{domain}'"
            for generic, domain in self.terminology_map.items()
        ]

        banned = self._all_banned_phrases()
        hedges = self._all_hedge_examples()

        if risk_percentage is not None:
            level_clause = f" ({risk_level})" if risk_level else ""
            risk_line = f"Estimated risk for this patient: {risk_percentage:.1f}%{level_clause}"
        else:
            risk_line = (
                f"Baseline prediction: {context.base_value:.4f}, "
                f"final prediction: {context.predicted_value:.4f}"
            )

        prompt_parts = [
            "You are writing a plain-language explanation of a health risk "
            "prediction for someone with no statistics or machine learning "
            "background. Write like a caring nurse explaining results in "
            "conversation, not like a technical report.",
            "",
            f"Tone profile: {self.tone_profile}",
            f"Target audience: {self.audience}",
            f"Target reading level: Flesch-Kincaid grade {target_grade} — "
            "keep sentences short and use everyday words.",
            "",
            "Use the following terminology substitutions where generic ML "
            "concepts appear:",
            *terminology_lines,
            "",
            risk_line,
            "",
            "Top contributing factors, strongest first:",
            *contribution_lines,
            "",
            "IMPORTANT — causal language restrictions:",
            "Do NOT use any of the following banned phrases:",
            ", ".join(f'"{p}"' for p in banned),
            "",
            "Instead, use hedging language such as:",
            ", ".join(f'"{h}"' for h in hedges),
            "",
            "IMPORTANT — avoid technical/statistical jargon entirely. Do NOT "
            "use any of these words or similar: " + ", ".join(f'"{j}"' for j in JARGON_TERMS) + ". "
            "Do not mention any raw numbers, scores, weights, or percentiles "
            "for individual factors — describe direction and importance in "
            "plain words only (e.g. 'the biggest factor was...', "
            "'this also played a role...').",
            "",
            "Write 6-10 short, warm, plain-English sentences covering all of "
            "the listed factors, explaining what most likely influenced this "
            "result and what that means for the patient in everyday terms. "
            "End with a brief, gentle note that this is a computer estimate, "
            "not a diagnosis, and does not prove any single factor causes "
            "the condition.",
        ]
        return "\n".join(prompt_parts)

    def explain(
        self,
        context: ExplanationContext,
        risk_percentage: Optional[float] = None,
        risk_level: Optional[str] = None,
    ) -> Narrative:
        """Generate a narrative for *context* via the configured LLM client.

        Parameters
        ----------
        context:
            Structured SHAP explanation for a single instance.
        risk_percentage:
            Optional real-world risk percentage to show instead of raw
            log-odds values. See :meth:`build_prompt`.
        risk_level:
            Optional risk level label (e.g. "MODERATE RISK").

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

        prompt = self.build_prompt(context, risk_percentage=risk_percentage, risk_level=risk_level)
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