"""Base formatter interface and shared stats extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from dfcontext.analyzers.base import ColumnSummary


@dataclass
class StatsBlock:
    """Intermediate representation of column statistics for formatting.

    Each formatter converts this into its own output format,
    avoiding duplicated extraction logic.
    """

    name: str
    column_type: str
    header_label: str
    lines: list[str] = field(default_factory=list)
    non_null_pct: int | None = None


def extract_stats_block(s: ColumnSummary) -> StatsBlock:
    """Extract formatting-agnostic stats from a ColumnSummary.

    Parameters
    ----------
    s : ColumnSummary
        The column summary to extract stats from.

    Returns
    -------
    StatsBlock
        Intermediate representation with label and content lines.

    """
    header: str = s.column_type
    if s.column_type == "categorical":
        header = f"categorical, {s.unique_count} unique"

    lines: list[str] = []
    stats = s.stats

    if s.column_type == "numeric":
        if "min" in stats and "max" in stats:
            lines.append(
                f"Range: {stats['min']:,.2f} — {stats['max']:,.2f}"
            )
        if "mean" in stats:
            lines.append(f"Mean: {stats['mean']:,.2f}")
        if "std" in stats:
            lines.append(f"Std: {stats['std']:,.2f}")
        if s.distribution_sketch:
            lines.append(f"Distribution: [{s.distribution_sketch}]")

    elif s.column_type == "categorical":
        top = stats.get("top_values", {})
        if top:
            lines.append(
                ", ".join(f"{k} ({v}%)" for k, v in top.items())
            )

    elif s.column_type == "text":
        if "avg_length" in stats:
            lines.append(
                f"Avg length: {stats['avg_length']} chars"
                f" (min: {stats.get('min_length', '?')},"
                f" max: {stats.get('max_length', '?')})"
            )
        patterns = stats.get("patterns", [])
        if patterns:
            lines.append(
                "Detected patterns: " + ", ".join(patterns)
            )

    elif s.column_type == "datetime":
        if "min" in stats and "max" in stats:
            line = f"Range: {stats['min']} — {stats['max']}"
            if "granularity" in stats:
                line += f" | Granularity: {stats['granularity']}"
            lines.append(line)

    elif s.column_type == "boolean":
        if "true_rate" in stats:
            tr = stats["true_rate"] * 100
            fr = stats["false_rate"] * 100
            lines.append(f"True: {tr:.1f}% | False: {fr:.1f}%")

    non_null_pct = None
    if s.non_null_rate < 1.0:
        non_null_pct = int(s.non_null_rate * 100)

    return StatsBlock(
        name=s.name,
        column_type=s.column_type,
        header_label=header,
        lines=lines,
        non_null_pct=non_null_pct,
    )


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
