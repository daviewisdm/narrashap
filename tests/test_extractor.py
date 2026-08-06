"""Tests for narrashap.core.extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from narrashap.core.extractor import extract, percentile_rank


@dataclass
class MockExplanation:
    """Minimal stand-in for shap.Explanation in unit tests."""

    values: Any
    base_values: Any
    data: Any
    feature_names: list[str]


TRAINING_DATA = pd.DataFrame(
    {
        "age": [20, 30, 40, 50, 60, np.nan, 70],
        "bmi": [18.0, 22.0, 25.0, 28.0, 30.0, 31.0, 35.0],
        "smoking": [0, 0, 1, 1, 0, 1, 1],
    }
)


class TestPercentileRank:
    def test_mid_distribution(self) -> None:
        column = pd.Series([10, 20, 30, 40, 50])
        assert percentile_rank(30, column) == 60.0

    def test_minimum_value(self) -> None:
        column = pd.Series([10, 20, 30, 40, 50])
        assert percentile_rank(10, column) == 20.0

    def test_maximum_value(self) -> None:
        column = pd.Series([10, 20, 30, 40, 50])
        assert percentile_rank(50, column) == 100.0

    def test_value_not_in_distribution(self) -> None:
        column = pd.Series([10, 20, 30, 40, 50])
        assert percentile_rank(25, column) == 40.0

    def test_nans_excluded(self) -> None:
        column = pd.Series([10, 20, np.nan, 30, np.nan, 40])
        # clean values: 10, 20, 30, 40 -> 30 is 75th percentile
        assert percentile_rank(30, column) == 75.0

    def test_numpy_array_input(self) -> None:
        column = np.array([1.0, 2.0, 3.0, 4.0])
        assert percentile_rank(2.0, column) == 50.0


class TestExtractWithExplanationObject:
    def test_extract_from_mock_explanation(self) -> None:
        explanation = MockExplanation(
            values=np.array([0.3, -0.1, 0.2]),
            base_values=np.array([0.5]),
            data=np.array([55, 28.0, 1]),
            feature_names=["age", "bmi", "smoking"],
        )

        context = extract(explanation, instance=None, training_data=TRAINING_DATA)

        assert context.base_value == pytest.approx(0.5)
        assert context.predicted_value == pytest.approx(0.5 + 0.3 - 0.1 + 0.2)
        assert len(context.contributions) == 3
        assert context.contributions[0].name == "age"
        assert context.contributions[0].shap_value == pytest.approx(0.3)
        assert context.contributions[0].percentile is not None
        assert context.contributions[1].name == "smoking"
        assert context.contributions[2].name == "bmi"


class TestExtractWithPlainArrays:
    def test_extract_from_arrays(self) -> None:
        shap_values = [0.1, -0.05, 0.2]
        instance = [45, 26.0, 0]
        feature_names = ["age", "bmi", "smoking"]

        context = extract(
            shap_values,
            instance,
            TRAINING_DATA,
            feature_names=feature_names,
            base_value=0.4,
        )

        assert context.base_value == pytest.approx(0.4)
        assert context.predicted_value == pytest.approx(0.4 + 0.1 - 0.05 + 0.2)
        assert [c.name for c in context.contributions] == ["smoking", "age", "bmi"]

    def test_missing_feature_names_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_names is required"):
            extract([0.1, 0.2], [1, 2], TRAINING_DATA, base_value=0.5)

    def test_missing_base_value_raises(self) -> None:
        with pytest.raises(ValueError, match="base_value is required"):
            extract(
                [0.1, 0.2],
                [1, 2],
                TRAINING_DATA,
                feature_names=["age", "bmi"],
            )


class TestExtractEdgeCases:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            extract(
                [0.1, 0.2],
                [1, 2, 3],
                TRAINING_DATA,
                feature_names=["age", "bmi"],
                base_value=0.5,
            )

    def test_feature_absent_from_training_data(self) -> None:
        shap_values = [0.15]
        instance = [100]
        feature_names = ["unknown_feature"]

        context = extract(
            shap_values,
            instance,
            TRAINING_DATA,
            feature_names=feature_names,
            base_value=0.2,
        )

        assert len(context.contributions) == 1
        assert context.contributions[0].name == "unknown_feature"
        assert context.contributions[0].percentile is None
