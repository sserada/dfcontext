"""Markdown output formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfcontext.formatters.base import BaseFormatter

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
        parts: list[str] = []
        for s in summaries:
            parts.append(
                _format_column_stats_md(s)
            )
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


def _format_column_stats_md(s: ColumnSummary) -> str:
    """Format a single column's statistics."""
    type_info: str = s.column_type
    if s.column_type == "categorical":
        type_info = f"categorical, {s.unique_count} unique"

    lines = [f"### {s.name} ({type_info})"]

    stats: dict[str, Any] = s.stats

    if s.column_type == "numeric":
        if "min" in stats and "max" in stats:
            line = f"Range: {stats['min']:,.2f} — {stats['max']:,.2f}"
            if "mean" in stats:
                line += f" | Mean: {stats['mean']:,.2f}"
            if "std" in stats:
                line += f" | Std: {stats['std']:,.2f}"
            lines.append(line)
        if s.distribution_sketch:
            lines.append(f"Distribution: [{s.distribution_sketch}]")

    elif s.column_type == "categorical":
        top = stats.get("top_values", {})
        if top:
            entries = [f"{k} ({v}%)" for k, v in top.items()]
            lines.append("Top values: " + ", ".join(entries))

    elif s.column_type == "text":
        if "avg_length" in stats:
            lines.append(
                f"Avg length: {stats['avg_length']} chars "
                f"(min: {stats.get('min_length', '?')}, "
                f"max: {stats.get('max_length', '?')})"
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

    # Non-null rate (if not 100%)
    if s.non_null_rate < 1.0:
        lines.append(f"Non-null: {s.non_null_rate * 100:.0f}%")

    return "\n".join(lines)
