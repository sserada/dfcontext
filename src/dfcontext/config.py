"""Configuration for context generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


_DEFAULT_BUDGET_RATIO: dict[str, float] = {
    "schema": 0.15,
    "stats": 0.55,
    "samples": 0.30,
}

_VALID_FORMATS = ("markdown", "plain", "yaml")


@dataclass
class ContextConfig:
    """Configuration for ``to_context()``.

    Parameters
    ----------
    token_budget : int
        Maximum number of tokens in the output.
    format : str
        Output format: ``"markdown"``, ``"plain"``, or ``"yaml"``.
    hint : str or None
        Query hint for prioritizing relevant columns.
    include_schema : bool
        Whether to include schema information.
    include_stats : bool
        Whether to include column statistics.
    include_samples : bool
        Whether to include sample rows.
    max_sample_rows : int
        Maximum number of sample rows.
    tokenizer : str
        tiktoken encoding name.
    budget_ratio : dict[str, float]
        Token budget allocation ratios for schema, stats, and samples.
    """

    token_budget: int = 2000
    format: Literal["markdown", "plain", "yaml"] = "markdown"
    hint: str | None = None
    include_schema: bool = True
    include_stats: bool = True
    include_samples: bool = True
    max_sample_rows: int = 5
    tokenizer: str = "cl100k_base"
    budget_ratio: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_BUDGET_RATIO)
    )

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.format not in _VALID_FORMATS:
            raise ValueError(
                f"format must be one of {_VALID_FORMATS}, got {self.format!r}"
            )
        if self.max_sample_rows < 0:
            raise ValueError("max_sample_rows must be non-negative")
