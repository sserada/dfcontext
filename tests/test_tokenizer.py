"""Tests for TokenCounter."""

from dfcontext.tokenizer import TokenCounter


class TestTokenCounterWithTiktoken:
    """Tests using tiktoken (installed in dev environment)."""

    def test_count_empty_string(self) -> None:
        tc = TokenCounter()
        assert tc.count("") == 0

    def test_count_simple_text(self) -> None:
        tc = TokenCounter()
        count = tc.count("hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_fits_within_budget(self) -> None:
        tc = TokenCounter()
        assert tc.fits("hello", 100)

    def test_fits_exceeds_budget(self) -> None:
        tc = TokenCounter()
        assert not tc.fits("hello world " * 100, 5)

    def test_truncate_within_budget(self) -> None:
        tc = TokenCounter()
        text = "hello world"
        assert tc.truncate(text, 100) == text

    def test_truncate_exceeds_budget(self) -> None:
        tc = TokenCounter()
        text = "hello world " * 100
        truncated = tc.truncate(text, 5)
        assert tc.count(truncated) <= 5

    def test_truncate_zero_budget(self) -> None:
        tc = TokenCounter()
        assert tc.truncate("hello", 0) == ""

    def test_truncate_negative_budget(self) -> None:
        tc = TokenCounter()
        assert tc.truncate("hello", -1) == ""

    def test_count_deterministic(self) -> None:
        tc = TokenCounter()
        text = "The quick brown fox jumps over the lazy dog."
        assert tc.count(text) == tc.count(text)
