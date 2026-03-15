"""Tests for PlainTextFormatter."""

import pandas as pd

from dfcontext.analyzers.base import ColumnSummary
from dfcontext.formatters.plain import PlainTextFormatter


class TestPlainTextFormatter:
    def test_format_schema(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        fmt = PlainTextFormatter()
        result = fmt.format_schema(df)

        assert "Dataset:" in result
        assert "2 rows" in result
        assert "a (int64" in result
        assert "##" not in result  # No markdown

    def test_format_stats(self) -> None:
        summary = ColumnSummary(
            name="price",
            dtype="float64",
            column_type="numeric",
            non_null_rate=1.0,
            unique_count=50,
            stats={"min": 1.0, "max": 100.0, "mean": 50.0},
        )
        fmt = PlainTextFormatter()
        result = fmt.format_stats([summary])

        assert "Statistics:" in result
        assert "price (numeric):" in result
        assert "Range:" in result

    def test_format_samples(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        fmt = PlainTextFormatter()
        result = fmt.format_samples(df, max_rows=2)

        assert "Sample rows:" in result
        assert "x=1" in result

    def test_format_samples_empty(self) -> None:
        df = pd.DataFrame()
        fmt = PlainTextFormatter()
        assert fmt.format_samples(df, max_rows=5) == ""
