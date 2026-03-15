"""Tests for to_context() core function."""

import numpy as np
import pandas as pd

from dfcontext import ContextConfig, to_context
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
        assert "2" in result  # 2 rows
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


class TestToContextBudget:
    def test_output_within_budget(self) -> None:
        df = pd.DataFrame({f"col_{i}": range(100) for i in range(20)})
        budget = 200
        result = to_context(df, token_budget=budget)
        tc = TokenCounter()
        assert tc.count(result) <= budget

    def test_small_budget_still_returns_something(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = to_context(df, token_budget=50)
        assert len(result) > 0


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
        assert "b" not in result.split("columns")[1]  # b not in schema rows


class TestToContextNullHandling:
    def test_dataframe_with_nulls(self) -> None:
        df = pd.DataFrame({"x": [1, None, 3], "y": [None, None, None]})
        result = to_context(df)
        assert isinstance(result, str)
        assert "67%" in result or "0%" in result  # non-null rates

    def test_all_null_dataframe(self) -> None:
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        result = to_context(df)
        assert isinstance(result, str)
        assert "0%" in result
