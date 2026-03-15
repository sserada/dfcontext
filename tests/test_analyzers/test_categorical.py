"""Tests for CategoricalAnalyzer."""

import pandas as pd

from dfcontext.analyzers.categorical import CategoricalAnalyzer


class TestCategoricalAnalyzer:
    def test_basic_analysis(self) -> None:
        s = pd.Series(["a", "b", "a", "c", "a"], name="cat")
        analyzer = CategoricalAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.name == "cat"
        assert summary.column_type == "categorical"
        assert summary.stats["unique_count"] == 3
        top = summary.stats["top_values"]
        assert "a" in top
        assert top["a"] == 60.0

    def test_with_nulls(self) -> None:
        s = pd.Series(["x", None, "x", None], name="c")
        analyzer = CategoricalAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.non_null_rate == 0.5
        assert summary.stats["top_values"]["x"] == 100.0

    def test_all_null(self) -> None:
        s = pd.Series([None, None], name="empty")
        analyzer = CategoricalAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats == {}
        assert summary.sample_values == []

    def test_top_n_scales_with_budget(self) -> None:
        values = [f"val_{i}" for i in range(50)] * 2
        s = pd.Series(values, name="many")
        analyzer = CategoricalAnalyzer()

        small = analyzer.analyze(s, budget=60)
        large = analyzer.analyze(s, budget=400)

        assert len(small.stats["top_values"]) <= len(
            large.stats["top_values"]
        )
