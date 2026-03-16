"""Tests for compare_contexts."""

import numpy as np
import pandas as pd

from dfcontext import compare_contexts, count_tokens


class TestCompareContexts:
    def test_basic_comparison(self) -> None:
        df_a = pd.DataFrame({"x": range(50)})
        df_b = pd.DataFrame({"x": range(100)})
        result = compare_contexts(df_a, df_b)

        assert "## Dataset A" in result
        assert "## Dataset B" in result
        assert "50 rows" in result
        assert "100 rows" in result

    def test_custom_labels(self) -> None:
        df_a = pd.DataFrame({"x": [1]})
        df_b = pd.DataFrame({"x": [2]})
        result = compare_contexts(
            df_a, df_b, label_a="2023", label_b="2024"
        )
        assert "## 2023" in result
        assert "## 2024" in result

    def test_budget_split(self) -> None:
        np.random.seed(42)
        df_a = pd.DataFrame({f"c{i}": range(100) for i in range(10)})
        df_b = pd.DataFrame({f"c{i}": range(100) for i in range(10)})
        result = compare_contexts(df_a, df_b, token_budget=1000)
        tokens = count_tokens(result)
        assert tokens <= 1100  # some overhead from labels

    def test_with_hint(self) -> None:
        df_a = pd.DataFrame({"sales": [100], "id": [1]})
        df_b = pd.DataFrame({"sales": [200], "id": [2]})
        result = compare_contexts(df_a, df_b, hint="sales")
        assert "sales" in result
