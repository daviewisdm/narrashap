"""LLM client abstractions for narrative generation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from narrashap.core.extractor import ExplanationContext


class LLMClient(Protocol):
    """Protocol for language-model backends used by narrators."""

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate text from a prompt."""
        ...


class AnthropicClient:
    """Anthropic Claude API client with lazy SDK import."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required: pass api_key or set ANTHROPIC_API_KEY"
            )

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Call the Anthropic Messages API and return concatenated text blocks."""
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicClient. "
                "Install with: pip install narrashap[anthropic]"
            ) from exc

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError as exc:
            # anthropic's exception classes require response=/body= kwargs in
            # their constructor, so they can't be re-raised with just a
            # string message. Wrap in a plain exception instead, preserving
            # the original error via `from exc`.
            raise RuntimeError(
                "Anthropic authentication failed. Verify your API key."
            ) from exc
        except anthropic.APIError as exc:
            raise RuntimeError(
                f"Anthropic API request failed: {exc}"
            ) from exc

        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)


class GroqClient:
    """Groq API client (OpenAI-compatible) with lazy SDK import.

    Free tier as of writing: 30 requests/min, 14,400/day on models like
    openai/gpt-oss-120b. No credit card required to start.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required: pass api_key or set GROQ_API_KEY"
            )

    def generate(self, prompt: str, max_tokens: int = 800) -> str:
        """Call the Groq chat completions API and return the response text."""
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "groq package is required for GroqClient. "
                "Install with: pip install groq"
            ) from exc

        try:
            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                reasoning_effort="low",  # keep more of the token budget for the actual narrative
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise RuntimeError(f"Groq API request failed: {exc}") from exc

        return response.choices[0].message.content or ""


class TemplateOnlyClient:
    """Template-based narrative generator that does not call an LLM."""

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Not supported — use :meth:`generate_from_context` instead."""
        raise NotImplementedError(
            "TemplateOnlyClient does not use prompt strings. "
            "Call generate_from_context(context, audience) instead."
        )

    def generate_from_context(
        self,
        context: ExplanationContext,
        audience: str,
        risk_percentage: Optional[float] = None,
        risk_level: Optional[str] = None,
    ) -> str:
        """Build a detailed, plain-language narrative purely from templates.

        Parameters
        ----------
        context:
            Structured SHAP explanation context.
        audience:
            Target audience label (used for light phrasing adjustments).
        risk_percentage:
            Optional value like 65.6 (meaning 65.6%). If provided, this is
            shown instead of the raw base_value/predicted_value, which are
            in log-odds space and not meaningful to a non-technical reader.
        risk_level:
            Optional label like "MODERATE RISK" to include in the summary.

        Returns
        -------
        str
            A longer, plain-language narrative covering the top
            contributing factors, written for a non-technical reader.
        """
        top = context.contributions[:5]  # was 3 — more detail

        if not top:
            return "There wasn't enough information to explain this prediction in detail."

        sentences: list[str] = []

        # Opening — plain language, no raw log-odds numbers shown to the reader
        if risk_percentage is not None:
            level_clause = f" This falls in the {risk_level.lower()} range." if risk_level else ""
            sentences.append(
                f"Based on the information provided, the estimated risk is "
                f"{risk_percentage:.1f}%.{level_clause} Here is what influenced that result the most."
            )
        else:
            direction_word = "went up" if context.predicted_value >= context.base_value else "went down"
            sentences.append(
                f"Starting from a general baseline, this patient's estimated risk {direction_word} "
                f"based on the factors below."
            )

        # Body — one plain sentence per top factor, strongest first
        for rank, contribution in enumerate(top, start=1):
            if contribution.shap_value > 0:
                effect = "raised"
            elif contribution.shap_value < 0:
                effect = "lowered"
            else:
                effect = "had little effect on"

            # Skip the percentile clause at the extremes — usually a sign
            # this is a yes/no or category feature, where "percentile"
            # isn't a meaningful idea to explain to a patient.
            percentile_clause = ""
            if contribution.percentile is not None and 0 < contribution.percentile < 100:
                percentile_clause = (
                    f" This is higher than about {contribution.percentile:.0f} out of "
                    f"100 women in our data."
                )

            rank_word = "The biggest factor was" if rank == 1 else "Another important factor was"
            sentences.append(
                f"{rank_word} {contribution.name.lower()}, which {effect} the estimated risk."
                f"{percentile_clause}"
            )

        # Closing — plain hedge against causal misreading
        if audience == "patient":
            sentences.append(
                "This is only a computer estimate based on patterns in data. It does not mean "
                "any one factor by itself causes fibroids, and it is not a diagnosis. "
                "Please talk to a doctor about what these results mean for you."
            )
        else:
            sentences.append(
                "These are statistical associations from the model, not proven causes. "
                "They should be interpreted alongside clinical judgment, not in place of it."
            )

        return " ".join(sentences)