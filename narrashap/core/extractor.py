"""Extract structured explanation context from SHAP values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureContribution:
    """A single feature's contribution to a model prediction."""

    name: str
    value: Any
    shap_value: float
    percentile: Optional[float] = None
    is_sensitive: bool = False


@dataclass
class ExplanationContext:
    """Structured context derived from a SHAP explanation for one instance."""

    base_value: float
    predicted_value: float
    contributions: list[FeatureContribution]
    instance_id: Optional[str] = None


def _is_shap_explanation(obj: Any) -> bool:
    """Return True if *obj* looks like a single-instance shap.Explanation."""
    return all(hasattr(obj, attr) for attr in ("values", "base_values", "data"))


def _normalize_shap_values(shap_values: Any) -> np.ndarray:
    """Flatten SHAP values to a 1-D float array."""
    arr = np.asarray(shap_values, dtype=float).ravel()
    return arr


def _normalize_instance(instance: Any) -> np.ndarray:
    """Flatten instance feature values to a 1-D array."""
    if isinstance(instance, pd.Series):
        return instance.values.ravel()
    return np.asarray(instance).ravel()


def _normalize_base_value(base_values: Any) -> float:
    """Extract a scalar base value from SHAP base_values."""
    arr = np.asarray(base_values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("base_values must not be empty")
    return float(arr[0])


def percentile_rank(value: float, column: Any) -> float:
    """Return the percentage of *column* values less than or equal to *value*.

    NaN entries in *column* are excluded from the computation. Uses only
    numpy/pandas (no scipy).

    Parameters
    ----------
    value:
        The value whose percentile rank is computed.
    column:
        A pandas Series, numpy array, or other array-like of reference values.

    Returns
    -------
    float
        Percentile rank in the range [0, 100].
    """
    if isinstance(column, pd.Series):
        clean = column.dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(column, dtype=float)
        clean = arr[~np.isnan(arr)]

    if clean.size == 0:
        return float("nan")

    return float(np.sum(clean <= value) / clean.size * 100.0)


def extract(
    shap_values: Any,
    instance: Any,
    training_data: pd.DataFrame,
    feature_names: Optional[list[str]] = None,
    *,
    base_value: Optional[float] = None,
) -> ExplanationContext:
    """Build an :class:`ExplanationContext` from SHAP output for one instance.

    Accepts either a ``shap.Explanation`` object or plain array-likes. When
    using plain arrays, *feature_names* and *base_value* must both be supplied.

    Parameters
    ----------
    shap_values:
        SHAP values for a single instance, or a ``shap.Explanation`` object.
    instance:
        Feature values for the instance being explained.
    training_data:
        Reference DataFrame used to compute feature percentiles.
    feature_names:
        Names for each feature. Required when *shap_values* is not a
        ``shap.Explanation`` object.
    base_value:
        Model baseline prediction. Required when *shap_values* is not a
        ``shap.Explanation`` object.

    Returns
    -------
    ExplanationContext
        Structured explanation with contributions sorted by absolute SHAP
        value descending.

    Raises
    ------
    ValueError
        If array inputs are missing required metadata or lengths mismatch.
    """
    if _is_shap_explanation(shap_values):
        explanation = shap_values
        values = _normalize_shap_values(explanation.values)
        instance_values = _normalize_instance(explanation.data)
        resolved_base = _normalize_base_value(explanation.base_values)
        names: list[str] = list(explanation.feature_names)
    else:
        if feature_names is None:
            raise ValueError(
                "feature_names is required when shap_values is not a shap.Explanation"
            )
        if base_value is None:
            raise ValueError(
                "base_value is required when shap_values is not a shap.Explanation"
            )
        values = _normalize_shap_values(shap_values)
        instance_values = _normalize_instance(instance)
        resolved_base = float(base_value)
        names = list(feature_names)

    if not (len(values) == len(instance_values) == len(names)):
        raise ValueError(
            "Length mismatch among shap_values, instance, and feature_names: "
            f"{len(values)}, {len(instance_values)}, {len(names)}"
        )

    predicted = resolved_base + float(np.sum(values))

    contributions: list[FeatureContribution] = []
    for name, feat_value, shap_val in zip(names, instance_values, values):
        percentile: Optional[float] = None
        if name in training_data.columns:
            percentile = percentile_rank(float(feat_value), training_data[name])

        contributions.append(
            FeatureContribution(
                name=name,
                value=feat_value,
                shap_value=float(shap_val),
                percentile=percentile,
            )
        )

    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    return ExplanationContext(
        base_value=resolved_base,
        predicted_value=predicted,
        contributions=contributions,
    )
