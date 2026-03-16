"""Text column analyzer."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from dfcontext.analyzers.base import BaseAnalyzer, ColumnSummary

if TYPE_CHECKING:
    import pandas as pd

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+"),
    "url": re.compile(r"https?://\S+"),
    "phone": re.compile(r"\+?\d[\d\-\s]{7,}\d"),
}


class TextAnalyzer(BaseAnalyzer):
    """Analyzer for free-text columns."""

    def analyze(self, series: pd.Series, budget: int) -> ColumnSummary:
        """Analyze a text column.

        Parameters
        ----------
        series : pd.Series
            Text column data.
        budget : int
            Token budget for this column.

        Returns
        -------
        ColumnSummary

        """
        info = self._base_info(series, "text")
        valid = series.dropna().astype(str)

        stats: dict[str, Any] = {}

        if len(valid) == 0:
            return ColumnSummary(**info, stats=stats, sample_values=[])

        lengths = valid.str.len()
        stats["avg_length"] = round(float(lengths.mean()), 1)
        stats["min_length"] = int(lengths.min())
        stats["max_length"] = int(lengths.max())

        # Pattern detection
        detected = _detect_patterns(valid)
        if detected:
            stats["patterns"] = detected

        # Representative samples
        n_samples = min(3, len(valid))
        samples = valid.head(n_samples).tolist()

        return ColumnSummary(**info, stats=stats, sample_values=samples)


def _detect_patterns(
    series: pd.Series,
    sample_size: int = 100,
) -> list[str]:
    """Detect common patterns in text data.

    Parameters
    ----------
    series : pd.Series
        Text data (NaN-free).
    sample_size : int
        Number of values to sample for pattern detection.

    Returns
    -------
    list[str]
        Names of detected patterns.

    """
    sample = series.head(sample_size)
    detected: list[str] = []
    for name, pattern in _PATTERNS.items():
        match_rate = sample.str.contains(pattern).mean()
        if match_rate > 0.3:
            detected.append(name)
    return detected
