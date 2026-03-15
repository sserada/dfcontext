"""Plain text output formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfcontext.formatters.base import BaseFormatter

if TYPE_CHECKING:
    import pandas as pd

    from dfcontext.analyzers.base import ColumnSummary


class PlainTextFormatter(BaseFormatter):
    """Format output as plain text."""

    def format_schema(self, df: pd.DataFrame) -> str:
        """Format schema as indented text."""
        rows, cols = df.shape
        lines = [
            f"Dataset: {rows:,} rows × {cols} columns",
            "",
            "Columns:",
        ]
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_pct = df[col].notna().mean() * 100
            lines.append(
                f"  {col} ({dtype}, {non_null_pct:.0f}% non-null)"
            )
        return "\n".join(lines)

    def format_stats(
        self, summaries: list[ColumnSummary]
    ) -> str:
        """Format column statistics as plain text."""
        parts: list[str] = []
        for s in summaries:
            parts.append(_format_column_stats_plain(s))
        if not parts:
            return ""
        return "Statistics:\n" + "\n\n".join(parts)

    def format_samples(
        self, df: pd.DataFrame, max_rows: int
    ) -> str:
        """Format sample rows as plain text."""
        if df.empty or max_rows <= 0:
            return ""

        sample = df.head(max_rows)
        lines = ["Sample rows:"]
        for idx, row in sample.iterrows():
            vals = ", ".join(
                f"{c}={row[c]}" for c in sample.columns
            )
            lines.append(f"  [{idx}] {vals}")
        return "\n".join(lines)


def _format_column_stats_plain(s: ColumnSummary) -> str:
    """Format a single column's statistics as plain text."""
    lines = [f"  {s.name} ({s.column_type}):"]
    stats: dict[str, Any] = s.stats

    if s.column_type == "numeric":
        if "min" in stats:
            lines.append(
                f"    Range: {stats['min']:,.2f} to "
                f"{stats['max']:,.2f}"
            )
        if "mean" in stats:
            lines.append(f"    Mean: {stats['mean']:,.2f}")
        if s.distribution_sketch:
            lines.append(
                f"    Distribution: [{s.distribution_sketch}]"
            )

    elif s.column_type == "categorical":
        top = stats.get("top_values", {})
        if top:
            for k, v in top.items():
                lines.append(f"    {k}: {v}%")

    elif s.column_type == "text":
        if "avg_length" in stats:
            lines.append(
                f"    Avg length: {stats['avg_length']} chars"
            )

    elif s.column_type == "datetime":
        if "min" in stats:
            lines.append(
                f"    Range: {stats['min']} to {stats['max']}"
            )
        if "granularity" in stats:
            lines.append(
                f"    Granularity: {stats['granularity']}"
            )

    elif s.column_type == "boolean":
        if "true_rate" in stats:
            lines.append(
                f"    True: {stats['true_rate'] * 100:.1f}%"
            )

    if s.non_null_rate < 1.0:
        lines.append(
            f"    Non-null: {s.non_null_rate * 100:.0f}%"
        )

    return "\n".join(lines)
