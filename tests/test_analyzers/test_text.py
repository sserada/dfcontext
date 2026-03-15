"""Tests for TextAnalyzer."""

import pandas as pd

from dfcontext.analyzers.text import TextAnalyzer


class TestTextAnalyzer:
    def test_basic_analysis(self) -> None:
        s = pd.Series(["hello world", "foo bar baz", "test"], name="txt")
        analyzer = TextAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.name == "txt"
        assert summary.column_type == "text"
        assert summary.stats["avg_length"] > 0
        assert summary.stats["min_length"] == 4
        assert summary.stats["max_length"] == 11

    def test_email_pattern_detected(self) -> None:
        s = pd.Series(
            ["user@example.com", "admin@test.org", "a@b.com"],
            name="email",
        )
        analyzer = TextAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert "email" in summary.stats.get("patterns", [])

    def test_url_pattern_detected(self) -> None:
        s = pd.Series(
            ["https://example.com", "http://test.org", "https://a.b"],
            name="urls",
        )
        analyzer = TextAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert "url" in summary.stats.get("patterns", [])

    def test_no_pattern_for_plain_text(self) -> None:
        s = pd.Series(["just", "plain", "text"], name="plain")
        analyzer = TextAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert "patterns" not in summary.stats

    def test_all_null(self) -> None:
        s = pd.Series([None, None], name="empty")
        analyzer = TextAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert summary.stats == {}
        assert summary.sample_values == []

    def test_sample_values(self) -> None:
        s = pd.Series(["a", "b", "c", "d", "e"], name="s")
        analyzer = TextAnalyzer()
        summary = analyzer.analyze(s, budget=200)

        assert len(summary.sample_values) == 3
