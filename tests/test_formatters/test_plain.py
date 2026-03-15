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
        assert "##" not in result

    def test_format_numeric_stats(self) -> None:
        summary = ColumnSummary(
            name="price",
            dtype="float64",
            column_type="numeric",
            non_null_rate=1.0,
            unique_count=50,
            stats={"min": 1.0, "max": 100.0, "mean": 50.0},
            distribution_sketch="▁▃▇█▅▂▁▁",
        )
        fmt = PlainTextFormatter()
        result = fmt.format_stats([summary])

        assert "Statistics:" in result
        assert "price (numeric):" in result
        assert "Range:" in result
        assert "Mean:" in result
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
        fmt = PlainTextFormatter()
        result = fmt.format_stats([summary])

        assert "region (categorical):" in result
        assert "East: 40.0%" in result
        assert "West: 35.0%" in result

    def test_format_text_stats(self) -> None:
        summary = ColumnSummary(
            name="note",
            dtype="object",
            column_type="text",
            non_null_rate=1.0,
            unique_count=100,
            stats={"avg_length": 25.3},
        )
        fmt = PlainTextFormatter()
        result = fmt.format_stats([summary])

        assert "note (text):" in result
        assert "Avg length: 25.3 chars" in result

    def test_format_datetime_stats(self) -> None:
        summary = ColumnSummary(
            name="ts",
            dtype="datetime64[ns]",
            column_type="datetime",
            non_null_rate=1.0,
            unique_count=100,
            stats={
                "min": "2024-01-01",
                "max": "2024-12-31",
                "granularity": "daily",
            },
        )
        fmt = PlainTextFormatter()
        result = fmt.format_stats([summary])

        assert "ts (datetime):" in result
        assert "Range: 2024-01-01 to 2024-12-31" in result
        assert "Granularity: daily" in result

    def test_format_boolean_stats(self) -> None:
        summary = ColumnSummary(
            name="flag",
            dtype="bool",
            column_type="boolean",
            non_null_rate=1.0,
            unique_count=2,
            stats={"true_rate": 0.75, "false_rate": 0.25},
        )
        fmt = PlainTextFormatter()
        result = fmt.format_stats([summary])

        assert "flag (boolean):" in result
        assert "True: 75.0%" in result

    def test_non_null_shown_when_below_100(self) -> None:
        summary = ColumnSummary(
            name="col",
            dtype="float64",
            column_type="numeric",
            non_null_rate=0.8,
            unique_count=10,
            stats={"min": 0.0, "max": 10.0, "mean": 5.0},
        )
        fmt = PlainTextFormatter()
        result = fmt.format_stats([summary])

        assert "Non-null: 80%" in result

    def test_format_empty_stats(self) -> None:
        fmt = PlainTextFormatter()
        assert fmt.format_stats([]) == ""

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
