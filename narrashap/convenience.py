"""One-line convenience wrapper: SHAP output straight to a narrative string."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from narrashap.core.extractor import ExplanationContext, extract
from narrashap.core.llm_client import LLMClient, TemplateOnlyClient
from narrashap.core.narrator import BaseNarrator
from narrashap.domains.fraud import FraudNarrator
from narrashap.domains.healthcare import HealthcareNarrator

# Registry of available domain narrators. Add new entries here as domains
# are implemented (e.g. "finance": FinanceNarrator) — narrate() itself
# never needs to change.
_DOMAIN_NARRATORS: dict[str, type[BaseNarrator]] = {
    "healthcare": HealthcareNarrator,
    "fraud": FraudNarrator,
    "generic": BaseNarrator,
}


@dataclass
class NarrateResult:
    """Full result from narrate(return_details=True)."""

    text: str
    fidelity_score: Optional[float]
    context: ExplanationContext


def narrate(
    shap_values: Any,
    instance: Any,
    training_data: Any,
    feature_names: Optional[list[str]] = None,
    *,
    base_value: Optional[float] = None,
    domain: str = "healthcare",
    audience: str = "patient",
    risk_percentage: Optional[float] = None,
    risk_level: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    return_details: bool = False,
):
    """Turn SHAP output into a plain-language narrative in one call.

    Zero-config by default: no API key required, no domain expertise
    needed to get a sensible result.

    Parameters
    ----------
    shap_values, instance, training_data, feature_names, base_value:
        Passed straight through to :func:`narrashap.core.extractor.extract`.
        See that function for accepted shapes (shap.Explanation object or
        plain arrays).
    domain:
        Which domain narrator to use: "healthcare", "fraud", or "generic".
        Defaults to "healthcare", currently the only domain validated
        against a real model. Ignored if llm_client is None (the
        zero-config template path doesn't vary by domain yet).
    audience:
        "patient", "clinician", or "executive". Defaults to "patient".
    risk_percentage, risk_level:
        Optional real-world display values (e.g. 65.6, "MODERATE RISK").
        Strongly recommended when your model's SHAP values are in
        log-odds space — without these, the narrative falls back to
        raw base/predicted values, which read as confusing jargon.
    llm_client:
        If None (default), uses TemplateOnlyClient — free, instant, no
        network call. Pass an AnthropicClient or GroqClient instance for
        richer LLM-generated prose.
    return_details:
        If True, returns a NarrateResult (text, fidelity_score, context)
        instead of a plain string. fidelity_score is None when using the
        default template path, since scoring only applies to LLM output.

    Returns
    -------
    str or NarrateResult
        The narrative text, or full result if return_details=True.

    Examples
    --------
    Zero-config, no API key:

    >>> from narrashap import narrate
    >>> text = narrate(shap_values, instance, training_data,
    ...                risk_percentage=65.6, risk_level="MODERATE RISK")
    >>> print(text)

    With a real LLM for richer prose:

    >>> from narrashap.core.llm_client import GroqClient
    >>> text = narrate(shap_values, instance, training_data,
    ...                risk_percentage=65.6, risk_level="MODERATE RISK",
    ...                llm_client=GroqClient())
    """
    context = extract(
        shap_values,
        instance,
        training_data,
        feature_names,
        base_value=base_value,
    )

    if llm_client is None:
        text = TemplateOnlyClient().generate_from_context(
            context,
            audience=audience,
            risk_percentage=risk_percentage,
            risk_level=risk_level,
        )
        fidelity_score = None
    else:
        narrator_cls = _DOMAIN_NARRATORS.get(domain)
        if narrator_cls is None:
            valid = ", ".join(sorted(_DOMAIN_NARRATORS))
            raise ValueError(f"Unknown domain '{domain}'. Valid domains: {valid}")

        narrator = narrator_cls(audience=audience, llm_client=llm_client)
        narrative = narrator.explain(
            context,
            risk_percentage=risk_percentage,
            risk_level=risk_level,
        )
        text = narrative.text
        fidelity_score = narrative.fidelity_score

    if return_details:
        return NarrateResult(text=text, fidelity_score=fidelity_score, context=context)
    return text