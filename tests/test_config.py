"""Tests for ContextConfig."""

import pytest

from dfcontext.config import ContextConfig


class TestContextConfigDefaults:
    def test_default_values(self) -> None:
        cfg = ContextConfig()
        assert cfg.token_budget == 2000
        assert cfg.format == "markdown"
        assert cfg.hint is None
        assert cfg.include_schema is True
        assert cfg.include_stats is True
        assert cfg.include_samples is True
        assert cfg.max_sample_rows == 5
        assert cfg.tokenizer == "cl100k_base"

    def test_default_budget_ratio(self) -> None:
        cfg = ContextConfig()
        assert cfg.budget_ratio == {
            "schema": 0.15,
            "stats": 0.55,
            "samples": 0.30,
        }

    def test_budget_ratio_is_independent_copy(self) -> None:
        cfg1 = ContextConfig()
        cfg2 = ContextConfig()
        cfg1.budget_ratio["schema"] = 0.99
        assert cfg2.budget_ratio["schema"] == 0.15


class TestContextConfigValidation:
    def test_zero_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ContextConfig(token_budget=0)

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ContextConfig(token_budget=-1)

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            ContextConfig(format="json")  # type: ignore[arg-type]

    def test_negative_max_sample_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="max_sample_rows"):
            ContextConfig(max_sample_rows=-1)


class TestContextConfigCustom:
    def test_custom_values(self) -> None:
        cfg = ContextConfig(
            token_budget=5000,
            format="plain",
            hint="sales trends",
            include_samples=False,
            max_sample_rows=10,
        )
        assert cfg.token_budget == 5000
        assert cfg.format == "plain"
        assert cfg.hint == "sales trends"
        assert cfg.include_samples is False
        assert cfg.max_sample_rows == 10
