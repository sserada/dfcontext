"""Numeric column analyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dfcontext.analyzers.base import BaseAnalyzer, ColumnSummary

if TYPE_CHECKING:
    import pandas as pd

_HIST_CHARS = "▁▂▃▄▅▆▇█"


class NumericAnalyzer(BaseAnalyzer):
    """Analyzer for numeric columns."""

    def analyze(self, series: pd.Series, budget: int) -> ColumnSummary:
        """Analyze a numeric column.

        Scales detail with budget:
        - Tier 1 (any): min, max, mean, histogram
        - Tier 2 (>50): +median, std, quartiles, zero rate
        - Tier 3 (>200): +percentiles (5th/95th), skewness, kurtosis

        Parameters
        ----------
        series : pd.Series
            Numeric column data.
        budget : int
            Token budget for this column.

        Returns
        -------
        ColumnSummary

        """
        info = self._base_info(series, "numeric")
        valid = series.dropna()

        stats: dict[str, object] = {}

        if len(valid) == 0:
            return ColumnSummary(
                **info,
                stats=stats,
                sample_values=[],
            )

        # Tier 1: always included
        stats["min"] = float(valid.min())
        stats["max"] = float(valid.max())
        stats["mean"] = float(valid.mean())

        # Tier 2: budget > 50
        if budget > 50:
            stats["median"] = float(valid.median())
            stats["std"] = float(valid.std())
            stats["q1"] = float(valid.quantile(0.25))
            stats["q3"] = float(valid.quantile(0.75))
            stats["zero_rate"] = float((valid == 0).mean())

        # Tier 3: budget > 200
        if budget > 200 and len(valid) >= 10:
            stats["p5"] = float(valid.quantile(0.05))
            stats["p95"] = float(valid.quantile(0.95))
            skew_val = valid.skew()
            skew = float(skew_val) if isinstance(skew_val, (int, float)) else 0.0
            stats["skewness"] = round(skew, 3)
            if abs(skew) > 1:
                stats["skew_label"] = (
                    "right-skewed" if skew > 0 else "left-skewed"
                )

        histogram = _mini_histogram(valid)
        samples = valid.head(5).tolist()

        return ColumnSummary(
            **info,
            stats=stats,
            sample_values=samples,
            distribution_sketch=histogram,
        )


def _mini_histogram(series: pd.Series, bins: int = 8) -> str:
    """Generate a mini histogram using Unicode block characters.

    Parameters
    ----------
    series : pd.Series
        Numeric data (NaN-free).
    bins : int
        Number of bins.

    Returns
    -------
    str
        A string like ``"▁▃▇█▅▂▁▁"``.

    """
    if len(series) == 0:
        return ""

    counts, _ = np.histogram(series, bins=bins)
    max_count = counts.max()

    if max_count == 0:
        return _HIST_CHARS[0] * bins

    normalized = counts / max_count
    max_idx = len(_HIST_CHARS) - 1
    indices = np.clip(
        (normalized * max_idx).astype(int), 0, max_idx
    )
    return "".join(_HIST_CHARS[i] for i in indices)
