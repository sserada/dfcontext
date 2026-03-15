"""Datetime column analyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfcontext.analyzers.base import BaseAnalyzer, ColumnSummary

if TYPE_CHECKING:
    import pandas as pd


class DatetimeAnalyzer(BaseAnalyzer):
    """Analyzer for datetime columns."""

    def analyze(self, series: pd.Series, budget: int) -> ColumnSummary:
        """Analyze a datetime column.

        Parameters
        ----------
        series : pd.Series
            Datetime column data.
        budget : int
            Token budget for this column.

        Returns
        -------
        ColumnSummary
        """
        info = self._base_info(series, "datetime")
        valid = series.dropna()

        stats: dict[str, Any] = {}

        if len(valid) == 0:
            return ColumnSummary(**info, stats=stats, sample_values=[])

        stats["min"] = str(valid.min())
        stats["max"] = str(valid.max())

        granularity = _estimate_granularity(valid)
        if granularity:
            stats["granularity"] = granularity

        samples = [str(v) for v in valid.head(3).tolist()]

        return ColumnSummary(**info, stats=stats, sample_values=samples)


def _estimate_granularity(series: pd.Series) -> str | None:
    """Estimate the time granularity of a datetime series.

    Parameters
    ----------
    series : pd.Series
        Datetime data (NaN-free), sorted not required.

    Returns
    -------
    str or None
        Estimated granularity label, or ``None`` if undetermined.
    """
    if len(series) < 2:
        return None

    sorted_series = series.sort_values()
    diffs = sorted_series.diff().dropna()

    if len(diffs) == 0:
        return None

    import pandas as pd_mod

    median_diff = pd_mod.Timedelta(diffs.median())
    seconds = median_diff.total_seconds()

    if seconds < 1:
        return "sub-second"
    if seconds < 60:
        return "second"
    if seconds < 3600:
        return "minute"
    if seconds < 86400:
        return "hourly"
    if seconds < 86400 * 7:
        return "daily"
    if seconds < 86400 * 25:
        return "weekly"
    if seconds < 86400 * 180:
        return "monthly"
    return "yearly"
