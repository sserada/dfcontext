"""Categorical column analyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfcontext.analyzers.base import BaseAnalyzer, ColumnSummary

if TYPE_CHECKING:
    import pandas as pd


class CategoricalAnalyzer(BaseAnalyzer):
    """Analyzer for categorical columns."""

    def analyze(self, series: pd.Series, budget: int) -> ColumnSummary:
        """Analyze a categorical column.

        Parameters
        ----------
        series : pd.Series
            Categorical column data.
        budget : int
            Token budget for this column.

        Returns
        -------
        ColumnSummary
        """
        info = self._base_info(series, "categorical")
        valid = series.dropna()

        stats: dict[str, Any] = {}

        if len(valid) == 0:
            return ColumnSummary(**info, stats=stats, sample_values=[])

        counts = valid.value_counts()
        total = len(valid)

        # Determine how many top values to show based on budget
        top_n = min(len(counts), max(3, budget // 20))
        top_values: dict[str, float] = {}
        for val, count in counts.head(top_n).items():
            pct = count / total * 100
            top_values[str(val)] = round(pct, 1)

        stats["top_values"] = top_values
        stats["unique_count"] = int(counts.shape[0])

        samples = valid.head(5).tolist()

        return ColumnSummary(**info, stats=stats, sample_values=samples)
