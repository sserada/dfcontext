"""Main entry point for context generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dfcontext.config import ContextConfig

if TYPE_CHECKING:
    import pandas as pd
from dfcontext.tokenizer import TokenCounter


def to_context(
    df: pd.DataFrame,
    token_budget: int = 2000,
    format: Literal["markdown", "plain", "yaml"] = "markdown",  # noqa: A002
    hint: str | None = None,
    include_schema: bool = True,
    include_stats: bool = True,
    include_samples: bool = True,
    max_sample_rows: int = 5,
    columns: list[str] | None = None,
    tokenizer: str = "cl100k_base",
    config: ContextConfig | None = None,
) -> str:
    """Generate LLM context from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    token_budget : int
        Maximum token count for the output.
    format : str
        Output format (``"markdown"``, ``"plain"``, ``"yaml"``).
    hint : str or None
        Query hint to prioritize relevant columns.
    include_schema : bool
        Include schema information.
    include_stats : bool
        Include column statistics.
    include_samples : bool
        Include sample rows.
    max_sample_rows : int
        Maximum sample rows to include.
    columns : list[str] or None
        Subset of columns to include. ``None`` means all.
    tokenizer : str
        tiktoken encoding name.
    config : ContextConfig or None
        Config object. If given, overrides individual keyword arguments.

    Returns
    -------
    str
        Context string for use with an LLM.
    """
    if config is not None:
        cfg = config
    else:
        cfg = ContextConfig(
            token_budget=token_budget,
            format=format,
            hint=hint,
            include_schema=include_schema,
            include_stats=include_stats,
            include_samples=include_samples,
            max_sample_rows=max_sample_rows,
            tokenizer=tokenizer,
        )

    if df.empty:
        return ""

    # Select columns
    if columns is not None:
        df = df[columns]

    tc = TokenCounter(cfg.tokenizer)
    parts: list[str] = []

    # Schema section
    if cfg.include_schema:
        parts.append(_build_schema(df))

    result = "\n\n".join(parts)

    # Ensure output fits within budget
    if not tc.fits(result, cfg.token_budget):
        result = tc.truncate(result, cfg.token_budget)

    return result


def _build_schema(df: pd.DataFrame) -> str:
    """Build the schema section describing shape, columns, and dtypes."""
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
        lines.append(f"| {col} | {dtype} | {non_null_pct:.0f}% |")

    return "\n".join(lines)
