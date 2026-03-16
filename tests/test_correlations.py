"""Tests for correlation detection."""

import numpy as np
import pandas as pd

from dfcontext.correlations import find_top_correlations, format_correlations


class TestFindTopCorrelations:
    def test_strong_positive_correlation(self) -> None:
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),
            "c": np.random.randn(100),
        })
        pairs = find_top_correlations(df)
        assert len(pairs) >= 1
        assert pairs[0][0] == "a"
        assert pairs[0][1] == "b"
        assert pairs[0][2] > 0.9

    def test_no_numeric_columns(self) -> None:
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        assert find_top_correlations(df) == []

    def test_single_numeric_column(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert find_top_correlations(df) == []

    def test_low_correlation_filtered(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })
        pairs = find_top_correlations(df, min_abs_corr=0.5)
        assert len(pairs) == 0

    def test_max_pairs_limit(self) -> None:
        df = pd.DataFrame({
            f"c{i}": np.arange(100) * (i + 1) for i in range(10)
        })
        pairs = find_top_correlations(df, max_pairs=3)
        assert len(pairs) <= 3


class TestFormatCorrelations:
    def test_format_pairs(self) -> None:
        pairs = [("sales", "quantity", 0.823), ("price", "cost", -0.654)]
        result = format_correlations(pairs)
        assert "## Correlations" in result
        assert "sales ↔ quantity" in result
        assert "strong positive" in result
        assert "moderate negative" in result

    def test_empty_pairs(self) -> None:
        assert format_correlations([]) == ""


class TestIntegration:
    def test_to_context_with_correlations(self) -> None:
        from dfcontext import to_context

        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),
            "c": np.random.randn(100),
        })
        ctx = to_context(df, include_correlations=True)
        assert "Correlations" in ctx
        assert "a ↔ b" in ctx
