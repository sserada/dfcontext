"""Column correlation detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def find_top_correlations(
    df: pd.DataFrame,
    max_pairs: int = 5,
    min_abs_corr: float = 0.3,
) -> list[tuple[str, str, float]]:
    """Find the most correlated numeric column pairs.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    max_pairs : int
        Maximum number of pairs to return.
    min_abs_corr : float
        Minimum absolute correlation to include.

    Returns
    -------
    list[tuple[str, str, float]]
        List of (col_a, col_b, correlation) sorted by |correlation| descending.

    """
    import pandas as pd

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return []

    corr_matrix = numeric_df.corr()
    pairs: list[tuple[str, str, float]] = []

    cols = list(corr_matrix.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            val = float(corr_matrix.loc[col_a, col_b])
            if pd.notna(val) and abs(val) >= min_abs_corr:
                pairs.append((col_a, col_b, round(val, 3)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:max_pairs]


def format_correlations(
    pairs: list[tuple[str, str, float]],
) -> str:
    """Format correlation pairs as a text block.

    Parameters
    ----------
    pairs : list[tuple[str, str, float]]
        Correlation pairs from ``find_top_correlations()``.

    Returns
    -------
    str
        Formatted correlation section.

    """
    if not pairs:
        return ""

    lines = ["## Correlations"]
    for col_a, col_b, corr in pairs:
        strength = _label(corr)
        lines.append(f"- {col_a} ↔ {col_b}: r={corr:+.3f} ({strength})")
    return "\n".join(lines)


def _label(corr: float) -> str:
    """Return a human-readable strength label."""
    abs_c = abs(corr)
    if abs_c >= 0.8:
        direction = "positive" if corr > 0 else "negative"
        return f"strong {direction}"
    if abs_c >= 0.5:
        direction = "positive" if corr > 0 else "negative"
        return f"moderate {direction}"
    direction = "positive" if corr > 0 else "negative"
    return f"weak {direction}"
