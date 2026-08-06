"""End-to-end healthcare demo using TemplateOnlyClient (no API key required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from narrashap.core.extractor import extract
from narrashap.core.llm_client import TemplateOnlyClient
from narrashap.domains.healthcare import HealthcareNarrator


def main() -> None:
    rng = np.random.default_rng(42)
    n = 200
    age = rng.integers(25, 75, size=n)
    bmi = rng.normal(27, 4, size=n).clip(18, 40)
    smoking = rng.integers(0, 2, size=n)
    exercise = rng.integers(0, 2, size=n)

    # Synthetic risk: higher age, bmi, smoking increase log-odds; exercise decreases
    log_odds = -2.0 + 0.03 * age + 0.05 * bmi + 0.8 * smoking - 0.4 * exercise
    y = (log_odds + rng.normal(0, 0.5, size=n) > 0).astype(int)

    X = pd.DataFrame(
        {"age": age, "bmi": bmi, "smoking": smoking, "exercise": exercise}
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    explanation = explainer(X.iloc[[0]])[0]

    context = extract(explanation, instance=None, training_data=X)

    template_client = TemplateOnlyClient()
    narrative_text = template_client.generate_from_context(
        context,
        audience="clinician",
    )

    print("=== Healthcare Demo (TemplateOnlyClient) ===")
    print(narrative_text)
    print()
    print(
        "To use LLM-generated narratives instead, swap TemplateOnlyClient for "
        "AnthropicClient(api_key=...) and call HealthcareNarrator(...).explain(context)."
    )


if __name__ == "__main__":
    main()
