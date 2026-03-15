"""Tests for DatetimeAnalyzer."""

import pandas as pd

from dfcontext.analyzers.datetime import DatetimeAnalyzer


class TestDatetimeAnalyzer:
    def test_basic_analysis(self) -> None:
        s = pd.Series(
            pd.date_range("2024-01-01", periods=10, freq="D"),
            name="date",
        )
        analyzer = DatetimeAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.name == "date"
        assert summary.column_type == "datetime"
        assert "2024-01-01" in summary.stats["min"]
        assert "2024-01-10" in summary.stats["max"]
        assert summary.stats["granularity"] == "daily"

    def test_hourly_granularity(self) -> None:
        s = pd.Series(
            pd.date_range("2024-01-01", periods=48, freq="h"),
            name="ts",
        )
        analyzer = DatetimeAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats["granularity"] == "hourly"

    def test_monthly_granularity(self) -> None:
        s = pd.Series(
            pd.date_range("2024-01-01", periods=12, freq="MS"),
            name="month",
        )
        analyzer = DatetimeAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats["granularity"] == "monthly"

    def test_all_null(self) -> None:
        s = pd.Series([None, None], dtype="datetime64[ns]", name="empty")
        analyzer = DatetimeAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats == {}
        assert summary.sample_values == []

    def test_single_value(self) -> None:
        s = pd.Series(
            pd.to_datetime(["2024-06-15"]),
            name="single",
        )
        analyzer = DatetimeAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert "2024-06-15" in summary.stats["min"]
        assert "granularity" not in summary.stats
