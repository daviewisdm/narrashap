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
    ) -> str:
        """Build a narrative purely from string templates (no LLM call).

        Parameters
        ----------
        context:
            Structured SHAP explanation context.
        audience:
            Target audience label (used for light phrasing adjustments).

        Returns
        -------
        str
            Template-generated narrative text.
        """
        top = context.contributions[:3]
        if not top:
            return (
                f"The model's baseline estimate was {context.base_value:.3f}. "
                f"The final estimate is {context.predicted_value:.3f}."
            )

        sentences: list[str] = []
        direction_word = "increased" if context.predicted_value >= context.base_value else "decreased"
        sentences.append(
            f"The model's baseline estimate was {context.base_value:.3f}, "
            f"and the final estimate {direction_word} to {context.predicted_value:.3f}."
        )

        for contribution in top:
            if contribution.shap_value > 0:
                effect = "increasing"
            elif contribution.shap_value < 0:
                effect = "decreasing"
            else:
                effect = "having little effect on"

            percentile_clause = ""
            if contribution.percentile is not None:
                percentile_clause = (
                    f" (at the {contribution.percentile:.0f}th percentile "
                    f"in the reference population)"
                )

            sentences.append(
                f"{contribution.name} was one of the strongest factors in this "
                f"prediction, {effect} the estimate"
                f"{percentile_clause}."
            )

        if audience == "patient":
            sentences.append(
                "This description reflects how the model weighed factors and "
                "does not prove cause and effect."
            )
        else:
            sentences.append(
                "These attributions describe model behavior and should not be "
                "interpreted as causal evidence."
            )

        return " ".join(sentences)