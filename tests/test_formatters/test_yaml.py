"""Tests for YAMLFormatter."""

import numpy as np
import pandas as pd

from dfcontext.analyzers.base import ColumnSummary
from dfcontext.formatters.yaml_fmt import YAMLFormatter, _convert


class TestYAMLFormatter:
    def test_format_schema(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        fmt = YAMLFormatter()
        result = fmt.format_schema(df)

        assert "rows: 2" in result
        assert "columns: 2" in result

    def test_format_stats(self) -> None:
        summary = ColumnSummary(
            name="sales",
            dtype="float64",
            column_type="numeric",
            non_null_rate=1.0,
            unique_count=100,
            stats={"min": 0.0, "max": 100.0},
            distribution_sketch="▁▃▇█",
        )
        fmt = YAMLFormatter()
        result = fmt.format_stats([summary])

        assert "statistics:" in result
        assert "sales:" in result
        assert "distribution:" in result

    def test_format_samples(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        fmt = YAMLFormatter()
        result = fmt.format_samples(df, max_rows=2)

        assert "samples:" in result

    def test_format_samples_empty(self) -> None:
        df = pd.DataFrame()
        fmt = YAMLFormatter()
        assert fmt.format_samples(df, max_rows=5) == ""

    def test_format_samples_zero_rows(self) -> None:
        df = pd.DataFrame({"x": [1]})
        fmt = YAMLFormatter()
        assert fmt.format_samples(df, max_rows=0) == ""


class TestConvert:
    def test_numpy_int(self) -> None:
        result = _convert(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self) -> None:
        result = _convert(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_bool(self) -> None:
        result = _convert(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_datetime(self) -> None:
        ts = pd.Timestamp("2024-01-01")
        result = _convert(ts)
        assert isinstance(result, str)
        assert "2024" in result

    def test_plain_string(self) -> None:
        assert _convert("hello") == "hello"
