"""Tests for classify_column."""

import numpy as np
import pandas as pd

from dfcontext.analyzers import classify_column


class TestClassifyColumn:
    def test_boolean(self) -> None:
        s = pd.Series([True, False, True])
        assert classify_column(s) == "boolean"

    def test_datetime(self) -> None:
        s = pd.Series(pd.date_range("2024-01-01", periods=5))
        assert classify_column(s) == "datetime"

    def test_numeric_int(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])
        assert classify_column(s) == "numeric"

    def test_numeric_float(self) -> None:
        s = pd.Series([1.1, 2.2, 3.3])
        assert classify_column(s) == "numeric"

    def test_categorical_low_cardinality(self) -> None:
        s = pd.Series(["a", "b", "a", "b", "a", "b", "a", "b"])
        assert classify_column(s) == "categorical"

    def test_text_high_cardinality(self) -> None:
        s = pd.Series([f"unique_{i}" for i in range(10)])
        assert classify_column(s) == "text"

    def test_empty_series(self) -> None:
        s = pd.Series([], dtype=object)
        result = classify_column(s)
        assert result in ("categorical", "text")

    def test_nullable_boolean(self) -> None:
        s = pd.Series([True, None, False], dtype="boolean")
        assert classify_column(s) == "boolean"

    def test_numeric_with_nulls(self) -> None:
        s = pd.Series([1, np.nan, 3])
        assert classify_column(s) == "numeric"
