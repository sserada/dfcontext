"""Main entry point for context generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dfcontext.analyzers.base import ColumnSummary, classify_column
from dfcontext.analyzers.boolean import BooleanAnalyzer
from dfcontext.analyzers.categorical import CategoricalAnalyzer
from dfcontext.analyzers.datetime import DatetimeAnalyzer
from dfcontext.analyzers.numeric import NumericAnalyzer
from dfcontext.analyzers.text import TextAnalyzer
from dfcontext.budget import TokenBudgetAllocator
from dfcontext.config import ContextConfig
from dfcontext.correlations import find_top_correlations, format_correlations
from dfcontext.formatters.markdown import MarkdownFormatter
from dfcontext.formatters.plain import PlainTextFormatter
from dfcontext.formatters.yaml_fmt import YAMLFormatter
from dfcontext.sampler import RepresentativeSampler
from dfcontext.tokenizer import TokenCounter

if TYPE_CHECKING:
    import pandas as pd

    from dfcontext.formatters.base import BaseFormatter

_ANALYZERS = {
    "numeric": NumericAnalyzer(),
    "categorical": CategoricalAnalyzer(),
    "text": TextAnalyzer(),
    "datetime": DatetimeAnalyzer(),
    "boolean": BooleanAnalyzer(),
}

_FORMATTERS: dict[str, BaseFormatter] = {
    "markdown": MarkdownFormatter(),
    "plain": PlainTextFormatter(),
    "yaml": YAMLFormatter(),
}


def to_context(
    df: pd.DataFrame,
    token_budget: int = 2000,
    format: Literal["markdown", "plain", "yaml"] = "markdown",
    hint: str | None = None,
    include_schema: bool = True,
    include_stats: bool = True,
    include_samples: bool = True,
    include_correlations: bool = False,
    max_sample_rows: int = 5,
    columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    column_priority: dict[str, float] | None = None,
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
    include_correlations : bool
        Include numeric column correlations.
    max_sample_rows : int
        Maximum sample rows to include.
    columns : list[str] or None
        Subset of columns to include. ``None`` means all.
    exclude_columns : list[str] or None
        Columns to exclude (e.g. sensitive data). Applied after ``columns``.
    column_priority : dict[str, float] or None
        Explicit weight multipliers per column for budget allocation.
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
            include_correlations=include_correlations,
            max_sample_rows=max_sample_rows,
            column_priority=column_priority,
            tokenizer=tokenizer,
        )

    if df.empty:
        return ""

    # Select / exclude columns
    if columns is not None:
        df = df[columns]
    if exclude_columns is not None:
        df = df.drop(
            columns=[c for c in exclude_columns if c in df.columns]
        )

    tc = TokenCounter(cfg.tokenizer)
    formatter = _FORMATTERS[cfg.format]

    # Budget allocation
    allocator = TokenBudgetAllocator(cfg.budget_ratio)
    plan = allocator.allocate(
        df,
        cfg.token_budget,
        hint=cfg.hint,
        include_schema=cfg.include_schema,
        include_stats=cfg.include_stats,
        include_samples=cfg.include_samples,
        column_priority=cfg.column_priority,
    )

    parts: list[str] = []

    # Schema section
    if cfg.include_schema:
        schema_text = formatter.format_schema(df)
        if not tc.fits(schema_text, plan.schema_budget):
            schema_text = tc.truncate(
                schema_text, plan.schema_budget
            )
        parts.append(schema_text)

    # Stats section
    if cfg.include_stats:
        summaries = _analyze_all_columns(df, plan.column_budgets)
        stats_text = formatter.format_stats(summaries)
        if stats_text:
            parts.append(stats_text)

    # Samples section — scale row count with available budget
    if cfg.include_samples and cfg.max_sample_rows > 0:
        # Estimate ~30 tokens per sample row as a baseline
        budget_rows = max(1, plan.sample_budget // 30)
        effective_rows = min(cfg.max_sample_rows, budget_rows)
        sampler = RepresentativeSampler()
        sample_df = sampler.sample(df, effective_rows)
        samples_text = formatter.format_samples(
            sample_df, effective_rows
        )
        if samples_text:
            parts.append(samples_text)

    # Correlations section
    if cfg.include_correlations:
        corr_pairs = find_top_correlations(df)
        corr_text = format_correlations(corr_pairs)
        if corr_text:
            parts.append(corr_text)

    result = "\n\n".join(parts)

    # Final budget enforcement
    if not tc.fits(result, cfg.token_budget):
        import warnings

        warnings.warn(
            f"Output ({tc.count(result)} tokens) exceeds token_budget "
            f"({cfg.token_budget}). Output has been truncated. "
            "Consider increasing token_budget or disabling sections.",
            UserWarning,
            stacklevel=2,
        )
        result = tc.truncate(result, cfg.token_budget)

    return result


def analyze_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, ColumnSummary]:
    """Analyze columns and return structured summaries.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    columns : list[str] or None
        Columns to analyze. ``None`` means all.

    Returns
    -------
    dict[str, ColumnSummary]
        Column name to summary mapping.

    """
    cols = columns or list(df.columns)
    result: dict[str, ColumnSummary] = {}
    for col in cols:
        col_type = classify_column(df[col])
        analyzer = _ANALYZERS.get(col_type, _ANALYZERS["text"])
        result[col] = analyzer.analyze(df[col], budget=200)
    return result


def count_tokens(
    text: str, tokenizer: str = "cl100k_base"
) -> int:
    """Count the number of tokens in text.

    Parameters
    ----------
    text : str
        The text to count tokens for.
    tokenizer : str
        tiktoken encoding name.

    Returns
    -------
    int
        Token count.

    """
    tc = TokenCounter(tokenizer)
    return tc.count(text)


def compare_contexts(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
    token_budget: int = 2000,
    hint: str | None = None,
    **kwargs: object,
) -> str:
    """Generate a comparison context from two DataFrames.

    Splits the token budget evenly and produces a side-by-side summary.

    Parameters
    ----------
    df_a : pd.DataFrame
        First DataFrame.
    df_b : pd.DataFrame
        Second DataFrame.
    label_a : str
        Label for the first dataset.
    label_b : str
        Label for the second dataset.
    token_budget : int
        Total token budget (split between both datasets).
    hint : str or None
        Query hint applied to both datasets.
    **kwargs : object
        Additional keyword arguments passed to ``to_context()``.

    Returns
    -------
    str
        Combined context string for comparison.

    """
    per_dataset = token_budget // 2
    ctx_a = to_context(df_a, token_budget=per_dataset, hint=hint, **kwargs)  # type: ignore[arg-type]
    ctx_b = to_context(df_b, token_budget=per_dataset, hint=hint, **kwargs)  # type: ignore[arg-type]
    return f"## {label_a}\n{ctx_a}\n\n## {label_b}\n{ctx_b}"


def _analyze_all_columns(
    df: pd.DataFrame,
    column_budgets: dict[str, int],
) -> list[ColumnSummary]:
    """Analyze all columns with allocated budgets."""
    summaries: list[ColumnSummary] = []
    for col in df.columns:
        col_str = str(col)
        budget = column_budgets.get(col_str, 50)
        col_type = classify_column(df[col])
        analyzer = _ANALYZERS.get(col_type, _ANALYZERS["text"])
        summaries.append(analyzer.analyze(df[col], budget))
    return summaries
