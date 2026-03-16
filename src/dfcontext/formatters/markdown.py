"""Markdown output formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dfcontext.formatters.base import BaseFormatter, StatsBlock, extract_stats_block

if TYPE_CHECKING:
    import pandas as pd

    from dfcontext.analyzers.base import ColumnSummary


def _escape_md(text: str) -> str:
    """Escape pipe characters for Markdown table cells."""
    return text.replace("|", "\\|")


class MarkdownFormatter(BaseFormatter):
    """Format output as Markdown."""

    def format_schema(self, df: pd.DataFrame) -> str:
        """Format schema as Markdown table."""
        rows, cols = df.shape
        lines = [
            "## Dataset overview",
            f"- {rows:,} rows × {cols} columns",
            "",
            "## Schema",
            "| Column | Type | Non-null |",
            "|--------|------|----------|",
        ]
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_pct = df[col].notna().mean() * 100
            lines.append(
                f"| {_escape_md(str(col))} | {dtype} "
                f"| {non_null_pct:.0f}% |"
            )
        return "\n".join(lines)

    def format_stats(
        self, summaries: list[ColumnSummary]
    ) -> str:
        """Format column statistics as Markdown sections."""
        parts = [
            _render_stats_md(extract_stats_block(s))
            for s in summaries
        ]
        if not parts:
            return ""
        return "## Column statistics\n" + "\n\n".join(parts)

    def format_samples(
        self, df: pd.DataFrame, max_rows: int
    ) -> str:
        """Format sample rows as Markdown table."""
        if df.empty or max_rows <= 0:
            return ""

        sample = df.head(max_rows)
        cols = list(sample.columns)

        header = (
            "| "
            + " | ".join(_escape_md(str(c)) for c in cols)
            + " |"
        )
        sep = "|" + "|".join("---" for _ in cols) + "|"
        rows: list[str] = []
        for _, row in sample.iterrows():
            vals = " | ".join(
                _escape_md(str(row[c])) for c in cols
            )
            rows.append(f"| {vals} |")

        lines = [
            "## Sample rows (diverse selection)",
            header,
            sep,
            *rows,
        ]
        return "\n".join(lines)


def _render_stats_md(block: StatsBlock) -> str:
    """Render a StatsBlock as Markdown."""
    lines = [f"### {block.name} ({block.header_label})"]

    # Combine numeric range + mean + std on one line for compactness
    if block.column_type == "numeric" and len(block.lines) >= 2:
        # Merge Range + Mean + Std into single line
        combined = block.lines[0]  # Range
        for ln in block.lines[1:]:
            if ln.startswith("Mean:") or ln.startswith("Std:"):
                combined += f" | {ln}"
            else:
                lines.append(combined)
                combined = ln
        lines.append(combined)
    elif block.column_type == "categorical" and block.lines:
        lines.append("Top values: " + block.lines[0])
        lines.extend(block.lines[1:])
    else:
        lines.extend(block.lines)

    if block.non_null_pct is not None:
        lines.append(f"Non-null: {block.non_null_pct}%")

    return "\n".join(lines)
