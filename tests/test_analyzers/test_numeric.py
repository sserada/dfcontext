"""Tests for NumericAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from dfcontext.analyzers.numeric import NumericAnalyzer, _mini_histogram


class TestNumericAnalyzer:
    def test_basic_analysis(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5], name="x")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.name == "x"
        assert summary.column_type == "numeric"
        assert summary.non_null_rate == 1.0
        assert summary.stats["mean"] == pytest.approx(3.0)
        assert summary.stats["min"] == pytest.approx(1.0)
        assert summary.stats["max"] == pytest.approx(5.0)

    def test_with_nulls(self) -> None:
        s = pd.Series([1, None, 3, None, 5], name="y")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.non_null_rate == pytest.approx(0.6)
        assert summary.stats["mean"] == pytest.approx(3.0)

    def test_all_null(self) -> None:
        s = pd.Series([np.nan, np.nan], name="z")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats == {}
        assert summary.sample_values == []

    def test_single_value(self) -> None:
        s = pd.Series([42], name="single")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats["min"] == 42
        assert summary.stats["max"] == 42

    def test_quartiles_included_with_large_budget(self) -> None:
        s = pd.Series(range(100), name="q")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert "q1" in summary.stats
        assert "q3" in summary.stats
        assert "zero_rate" in summary.stats

    def test_quartiles_excluded_with_small_budget(self) -> None:
        s = pd.Series(range(100), name="q")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=30)

        assert "q1" not in summary.stats

    def test_tier3_with_high_budget(self) -> None:
        s = pd.Series(np.random.exponential(100, 1000), name="exp")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=300)

        assert "p5" in summary.stats
        assert "p95" in summary.stats
        assert "skewness" in summary.stats
        assert "skew_label" in summary.stats  # exponential is right-skewed

    def test_tier3_excluded_with_low_budget(self) -> None:
        s = pd.Series(range(100), name="lin")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=100)

        assert "p5" not in summary.stats
        assert "skewness" not in summary.stats

    def test_distribution_sketch(self) -> None:
        s = pd.Series(range(1000), name="d")
        analyzer = NumericAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.distribution_sketch is not None
        assert len(summary.distribution_sketch) == 8


class TestMiniHistogram:
    def test_uniform_distribution(self) -> None:
        s = pd.Series(range(1000))
        hist = _mini_histogram(s)
        assert len(hist) == 8

    def test_empty_series(self) -> None:
        s = pd.Series([], dtype=float)
        assert _mini_histogram(s) == ""

    def test_single_value_series(self) -> None:
        s = pd.Series([5, 5, 5, 5])
        hist = _mini_histogram(s)
        assert len(hist) == 8
