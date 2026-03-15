"""Tests for BooleanAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from dfcontext.analyzers.boolean import BooleanAnalyzer


class TestBooleanAnalyzer:
    def test_basic_analysis(self) -> None:
        s = pd.Series([True, False, True, True], name="flag")
        analyzer = BooleanAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.name == "flag"
        assert summary.column_type == "boolean"
        assert summary.stats["true_rate"] == pytest.approx(0.75)
        assert summary.stats["false_rate"] == pytest.approx(0.25)

    def test_all_true(self) -> None:
        s = pd.Series([True, True, True], name="all_true")
        analyzer = BooleanAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats["true_rate"] == 1.0
        assert summary.stats["false_rate"] == 0.0

    def test_with_nulls(self) -> None:
        s = pd.Series([True, None, False, None], name="nullable")
        analyzer = BooleanAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.non_null_rate == pytest.approx(0.5)
        assert summary.stats["true_rate"] == pytest.approx(0.5)

    def test_all_null(self) -> None:
        s = pd.Series(
            [np.nan, np.nan], dtype="boolean", name="empty"
        )
        analyzer = BooleanAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats == {}
