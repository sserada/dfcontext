"""Boolean column analyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfcontext.analyzers.base import BaseAnalyzer, ColumnSummary

if TYPE_CHECKING:
    import pandas as pd


class BooleanAnalyzer(BaseAnalyzer):
    """Analyzer for boolean columns."""

    def analyze(self, series: pd.Series, budget: int) -> ColumnSummary:
        """Analyze a boolean column.

        Parameters
        ----------
        series : pd.Series
            Boolean column data.
        budget : int
            Token budget for this column.

        Returns
        -------
        ColumnSummary

        """
        info = self._base_info(series, "boolean")
        valid = series.dropna()

        stats: dict[str, Any] = {}

        if len(valid) == 0:
            return ColumnSummary(**info, stats=stats, sample_values=[])

        true_rate = float(valid.mean())
        stats["true_rate"] = round(true_rate, 4)
        stats["false_rate"] = round(1 - true_rate, 4)

        return ColumnSummary(**info, stats=stats, sample_values=[])
