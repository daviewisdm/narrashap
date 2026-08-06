"""Fraud-detection-domain narrator with investigative terminology."""

from __future__ import annotations

from narrashap.core.narrator import BaseNarrator


class FraudNarrator(BaseNarrator):
    """Narrator tuned for fraud risk investigations.

    Terminology examples (generic -> domain):
    - feature -> signal
    - prediction -> fraud risk score
    - model -> fraud detection model
    - contribution -> influence on fraud score
    - baseline -> typical transaction profile
    - instance -> transaction or account profile
    - shap value -> signal weight
    - input -> transaction attribute
    """

    terminology_map: dict[str, str] = {
        "feature": "signal",
        "prediction": "fraud risk score",
        "model": "fraud detection model",
        "contribution": "influence on fraud score",
        "baseline": "typical transaction profile",
        "instance": "transaction profile",
        "shap value": "signal weight",
        "input": "transaction attribute",
        "predicted value": "fraud risk score",
        "base value": "baseline fraud rate",
        "attribution": "signal attribution",
        "score": "fraud score",
    }

    causal_language_policy: dict[str, list[str]] = {
        "banned": [
            "was fraudulent",
            "committed fraud",
            "is a fraudster",
            "guilty of fraud",
            "definitely fraud",
            "proven fraud",
        ],
        "required_hedges": [
            "may indicate elevated fraud risk",
            "appears to contribute to the fraud score",
            "is associated with the model's fraud assessment",
            "was among the signals influencing the score",
        ],
    }

    tone_profile: str = "investigative"
