"""Base analyzer and column classification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd


ColumnType = Literal["numeric", "categorical", "text", "datetime", "boolean"]


@dataclass
class ColumnSummary:
    """Summary of a single column's analysis.

    Parameters
    ----------
    name : str
        Column name.
    dtype : str
        pandas dtype as string.
    column_type : ColumnType
        Detected column type.
    non_null_rate : float
        Fraction of non-null values (0.0 to 1.0).
    unique_count : int
        Number of unique values.
    stats : dict[str, Any]
        Type-specific statistics.
    sample_values : list[Any]
        Representative sample values.
    distribution_sketch : str or None
        Mini histogram for numeric columns.

    """

    name: str
    dtype: str
    column_type: ColumnType
    non_null_rate: float
    unique_count: int
    stats: dict[str, Any] = field(default_factory=dict)
    sample_values: list[Any] = field(default_factory=list)
    distribution_sketch: str | None = None


class BaseAnalyzer(ABC):
    """Abstract base for column analyzers."""

    @abstractmethod
    def analyze(
        self, series: pd.Series, budget: int
    ) -> ColumnSummary:
        """Analyze a column and return a summary.

        Parameters
        ----------
        series : pd.Series
            The column data.
        budget : int
            Token budget allocated for this column's summary.

        Returns
        -------
        ColumnSummary

        """

    def _base_info(
        self,
        series: pd.Series,
        column_type: ColumnType,
    ) -> dict[str, Any]:
        """Compute common column metadata."""
        return {
            "name": str(series.name),
            "dtype": str(series.dtype),
            "column_type": column_type,
            "non_null_rate": float(series.notna().mean()),
            "unique_count": int(series.nunique()),
        }


def classify_column(
    series: pd.Series,
) -> ColumnType:
    """Classify a pandas Series into a column type.

    Parameters
    ----------
    series : pd.Series
        The column to classify.

    Returns
    -------
    ColumnType

    """
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    unique_ratio = series.nunique() / max(len(series), 1)
    if unique_ratio < 0.5:
        return "categorical"
    return "text"
