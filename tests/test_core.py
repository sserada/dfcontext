"""Tests for to_context() core function."""

import numpy as np
import pandas as pd

from dfcontext import (
    ContextConfig,
    analyze_columns,
    count_tokens,
    to_context,
)
from dfcontext.tokenizer import TokenCounter


class TestToContextBasic:
    def test_returns_string(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = to_context(df)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame()
        assert to_context(df) == ""

    def test_contains_shape_info(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = to_context(df)
        assert "2" in result
        assert "2 columns" in result

    def test_contains_column_names(self) -> None:
        df = pd.DataFrame({"price": [10.0], "name": ["item"]})
        result = to_context(df)
        assert "price" in result
        assert "name" in result

    def test_contains_dtypes(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = to_context(df)
        assert "int" in result.lower()


class TestToContextIntegration:
    def test_includes_statistics(self) -> None:
        df = pd.DataFrame({
            "region": ["East", "West", "East", "North"],
            "sales": [100.0, 200.0, 150.0, 300.0],
        })
        result = to_context(df, token_budget=2000)
        assert "Column statistics" in result
        assert "sales" in result

    def test_includes_samples(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        result = to_context(df, token_budget=2000)
        assert "Sample rows" in result

    def test_plain_format(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = to_context(df, format="plain")
        assert "##" not in result
        assert "Dataset:" in result

    def test_yaml_format(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = to_context(df, format="yaml")
        assert "rows:" in result or "columns:" in result

    def test_with_hint(self) -> None:
        df = pd.DataFrame({
            "sales": [100, 200, 300],
            "region": ["E", "W", "N"],
        })
        result = to_context(df, hint="sales analysis")
        assert "sales" in result

    def test_mixed_types_dataframe(self) -> None:
        df = pd.DataFrame({
            "id": range(50),
            "name": [f"item_{i}" for i in range(50)],
            "price": np.random.exponential(100, 50),
            "category": np.random.choice(["A", "B", "C"], 50),
            "active": np.random.choice([True, False], 50),
            "created": pd.date_range("2024-01-01", periods=50),
        })
        result = to_context(df, token_budget=2000)
        assert isinstance(result, str)
        tc = TokenCounter()
        assert tc.count(result) <= 2000

    def test_disable_sections(self) -> None:
        df = pd.DataFrame({"x": range(10)})
        result = to_context(
            df,
            include_schema=False,
            include_stats=False,
            include_samples=True,
        )
        assert "Dataset overview" not in result
        assert "Column statistics" not in result


class TestToContextBudget:
    def test_output_within_budget(self) -> None:
        df = pd.DataFrame(
            {f"col_{i}": range(100) for i in range(20)}
        )
        budget = 200
        result = to_context(df, token_budget=budget)
        tc = TokenCounter()
        assert tc.count(result) <= budget

    def test_small_budget_still_returns_something(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = to_context(df, token_budget=50)
        assert len(result) > 0

    def test_large_dataframe_within_budget(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            f"col_{i}": np.random.randn(10000) for i in range(10)
        })
        budget = 1000
        result = to_context(df, token_budget=budget)
        tc = TokenCounter()
        assert tc.count(result) <= budget


class TestToContextConfig:
    def test_accepts_config_object(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        cfg = ContextConfig(token_budget=1000)
        result = to_context(df, config=cfg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_column_selection(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = to_context(df, columns=["a", "c"])
        assert "a" in result
        assert "c" in result


class TestToContextNullHandling:
    def test_dataframe_with_nulls(self) -> None:
        df = pd.DataFrame(
            {"x": [1, None, 3], "y": [None, None, None]}
        )
        result = to_context(df)
        assert isinstance(result, str)

    def test_all_null_dataframe(self) -> None:
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        result = to_context(df)
        assert isinstance(result, str)
        assert "0%" in result


class TestAnalyzeColumns:
    def test_returns_summaries(self) -> None:
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "cat": ["a", "b", "a"],
        })
        result = analyze_columns(df)
        assert "num" in result
        assert "cat" in result
        assert result["num"].column_type == "numeric"

    def test_column_selection(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = analyze_columns(df, columns=["a", "c"])
        assert "a" in result
        assert "c" in result
        assert "b" not in result


class TestCountTokens:
    def test_basic_count(self) -> None:
        count = count_tokens("hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_empty_string(self) -> None:
        assert count_tokens("") == 0
