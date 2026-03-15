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

        stats["min"] = float(valid.min())
        stats["max"] = float(valid.max())
        stats["mean"] = float(valid.mean())
        stats["median"] = float(valid.median())
        stats["std"] = float(valid.std())

        if budget > 50:
            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            stats["q1"] = q1
            stats["q3"] = q3
            stats["zero_rate"] = float((valid == 0).mean())

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
