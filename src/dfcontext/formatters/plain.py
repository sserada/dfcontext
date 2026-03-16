"""Plain text output formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dfcontext.formatters.base import BaseFormatter, StatsBlock, extract_stats_block

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
        parts = [
            _render_stats_plain(extract_stats_block(s))
            for s in summaries
        ]
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


def _render_stats_plain(block: StatsBlock) -> str:
    """Render a StatsBlock as plain text."""
    lines = [f"  {block.name} ({block.column_type}):"]

    if block.column_type == "categorical":
        # Show each value on its own line
        for ln in block.lines:
            for entry in ln.split(", "):
                lines.append(f"    {entry}")
    else:
        for ln in block.lines:
            lines.append(f"    {ln}")

    if block.non_null_pct is not None:
        lines.append(f"    Non-null: {block.non_null_pct}%")

    return "\n".join(lines)
