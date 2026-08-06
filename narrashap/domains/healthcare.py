"""Healthcare-domain narrator with clinical terminology and guardrails."""

from __future__ import annotations

from narrashap.core.narrator import BaseNarrator


class HealthcareNarrator(BaseNarrator):
    """Narrator tuned for clinical risk explanations.

    Terminology examples (generic -> domain):
    - feature -> risk factor
    - prediction -> estimated risk
    - model -> clinical decision support tool
    - contribution -> influence on estimated risk
    - baseline -> population baseline risk
    - instance -> patient profile
    - shap value -> attribution weight
    - input -> clinical characteristic
    """

    terminology_map: dict[str, str] = {
        "feature": "risk factor",
        "prediction": "estimated risk",
        "model": "clinical decision support tool",
        "contribution": "influence on estimated risk",
        "baseline": "population baseline risk",
        "instance": "patient profile",
        "shap value": "attribution weight",
        "input": "clinical characteristic",
        "predicted value": "estimated patient risk",
        "base value": "baseline population risk",
        "attribution": "risk attribution",
        "score": "risk estimate",
    }

    causal_language_policy: dict[str, list[str]] = {
        "banned": [
            "will develop",
            "will get sick",
            "diagnosed with",
            "proves the patient has",
            "definitely has",
            "will suffer from",
        ],
        "required_hedges": [
            "may be associated with elevated risk",
            "appears to contribute to the estimated risk",
            "is linked to the model's risk estimate",
            "was among the factors influencing the estimate",
        ],
    }

    tone_profile: str = "clinical"
