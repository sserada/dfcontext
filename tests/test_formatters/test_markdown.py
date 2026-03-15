"""Tests for MarkdownFormatter."""

import pandas as pd

from dfcontext.analyzers.base import ColumnSummary
from dfcontext.formatters.markdown import MarkdownFormatter


class TestMarkdownFormatter:
    def test_format_schema(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        fmt = MarkdownFormatter()
        result = fmt.format_schema(df)

        assert "## Dataset overview" in result
        assert "2 rows" in result
        assert "2 columns" in result
        assert "| a |" in result
        assert "| b |" in result

    def test_format_numeric_stats(self) -> None:
        summary = ColumnSummary(
            name="sales",
            dtype="float64",
            column_type="numeric",
            non_null_rate=1.0,
            unique_count=100,
            stats={"min": 0.0, "max": 100.0, "mean": 50.0, "std": 25.0},
            distribution_sketch="▁▃▇█▅▂▁▁",
        )
        fmt = MarkdownFormatter()
        result = fmt.format_stats([summary])

        assert "## Column statistics" in result
        assert "sales" in result
        assert "Range:" in result
        assert "▁▃▇█▅▂▁▁" in result

    def test_format_categorical_stats(self) -> None:
        summary = ColumnSummary(
            name="region",
            dtype="object",
            column_type="categorical",
            non_null_rate=1.0,
            unique_count=3,
            stats={"top_values": {"East": 40.0, "West": 35.0}},
        )
        fmt = MarkdownFormatter()
        result = fmt.format_stats([summary])

        assert "categorical, 3 unique" in result
        assert "East (40.0%)" in result

    def test_format_samples(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        fmt = MarkdownFormatter()
        result = fmt.format_samples(df, max_rows=2)

        assert "## Sample rows" in result
        assert "| a | b |" in result
        assert "| 1 | x |" in result
        assert "3" not in result  # Only 2 rows

    def test_format_samples_empty(self) -> None:
        df = pd.DataFrame()
        fmt = MarkdownFormatter()
        assert fmt.format_samples(df, max_rows=5) == ""

    def test_non_null_shown_when_below_100(self) -> None:
        summary = ColumnSummary(
            name="col",
            dtype="float64",
            column_type="numeric",
            non_null_rate=0.75,
            unique_count=10,
            stats={"min": 0.0, "max": 10.0, "mean": 5.0, "std": 2.0},
        )
        fmt = MarkdownFormatter()
        result = fmt.format_stats([summary])

        assert "Non-null: 75%" in result

    def test_pipe_in_column_name_escaped(self) -> None:
        df = pd.DataFrame({"col|pipe": [1, 2], "normal": [3, 4]})
        fmt = MarkdownFormatter()
        schema = fmt.format_schema(df)

        assert "col\\|pipe" in schema
        assert schema.count("|") > 0  # Table structure intact

    def test_pipe_in_values_escaped(self) -> None:
        df = pd.DataFrame({"x": ["a|b", "c|d"]})
        fmt = MarkdownFormatter()
        result = fmt.format_samples(df, max_rows=2)

        assert "a\\|b" in result
