"""Base formatter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from dfcontext.analyzers.base import ColumnSummary


class BaseFormatter(ABC):
    """Abstract base for output formatters."""

    @abstractmethod
    def format_schema(
        self, df: pd.DataFrame
    ) -> str:
        """Format the schema section."""

    @abstractmethod
    def format_stats(
        self, summaries: list[ColumnSummary]
    ) -> str:
        """Format column statistics."""

    @abstractmethod
    def format_samples(
        self, df: pd.DataFrame, max_rows: int
    ) -> str:
        """Format sample rows."""
