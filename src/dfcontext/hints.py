"""Query hint analysis for column relevance scoring."""

from __future__ import annotations

from typing import Any


def compute_hint_relevance(
    hint: str,
    column_name: str,
    sample_values: list[Any] | None = None,
) -> float:
    """Compute relevance between a query hint and a column.

    Uses keyword matching (no LLM calls) to estimate how relevant
    a column is to the user's analysis intent.

    Parameters
    ----------
    hint : str
        The user's query hint.
    column_name : str
        The column name to score.
    sample_values : list[Any] or None
        Sample values from the column (useful for categorical).

    Returns
    -------
    float
        Relevance score between 0.0 and 1.0.
    """
    score = 0.0
    hint_lower = hint.lower()
    col_lower = column_name.lower()

    # Column name appears in hint
    if col_lower in hint_lower:
        score += 0.8

    # Hint keywords appear in column name
    hint_words = hint_lower.split()
    for word in hint_words:
        if len(word) > 2 and word in col_lower:
            score += 0.4

    # Sample values appear in hint (for categorical columns)
    if sample_values:
        for val in sample_values[:5]:
            if str(val).lower() in hint_lower:
                score += 0.2

    return min(score, 1.0)
