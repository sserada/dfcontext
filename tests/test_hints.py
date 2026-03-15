"""Tests for query hint analysis."""

from dfcontext.hints import compute_hint_relevance


class TestComputeHintRelevance:
    def test_exact_column_name_match(self) -> None:
        score = compute_hint_relevance("analyze sales trends", "sales")
        assert score >= 0.8

    def test_keyword_in_column_name(self) -> None:
        score = compute_hint_relevance("total sales by region", "region")
        assert score >= 0.4

    def test_no_match(self) -> None:
        score = compute_hint_relevance("analyze sales", "timestamp")
        assert score == 0.0

    def test_sample_value_match(self) -> None:
        score = compute_hint_relevance(
            "focus on East region",
            "area",
            sample_values=["East", "West", "North"],
        )
        assert score >= 0.2

    def test_score_capped_at_1(self) -> None:
        score = compute_hint_relevance(
            "sales sales sales",
            "sales",
            sample_values=["sales", "sales"],
        )
        assert score == 1.0

    def test_short_words_ignored(self) -> None:
        score = compute_hint_relevance("do it by id", "identifier")
        # "do", "it", "by" are <= 2 chars, should not match
        assert score < 0.4

    def test_case_insensitive(self) -> None:
        score = compute_hint_relevance("Sales Trends", "sales")
        assert score >= 0.8
