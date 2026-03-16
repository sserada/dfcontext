"""Token budget allocation across sections and columns."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dfcontext.hints import compute_hint_relevance

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BudgetPlan:
    """Token budget allocation plan.

    Parameters
    ----------
    schema_budget : int
        Tokens allocated to the schema section.
    column_budgets : dict[str, int]
        Tokens allocated per column for statistics.
    sample_budget : int
        Tokens allocated to sample rows.
    total_budget : int
        Total token budget.

    """

    schema_budget: int
    column_budgets: dict[str, int] = field(default_factory=dict)
    sample_budget: int = 0
    total_budget: int = 0


class TokenBudgetAllocator:
    """Allocate token budget across sections and columns.

    Parameters
    ----------
    budget_ratio : dict[str, float]
        Ratios for schema, stats, and samples sections.

    """

    def __init__(
        self,
        budget_ratio: dict[str, float] | None = None,
    ) -> None:
        self._ratio = budget_ratio or {
            "schema": 0.15,
            "stats": 0.55,
            "samples": 0.30,
        }

    def allocate(
        self,
        df: pd.DataFrame,
        total_budget: int,
        hint: str | None = None,
        include_schema: bool = True,
        include_stats: bool = True,
        include_samples: bool = True,
    ) -> BudgetPlan:
        """Create a budget plan for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to allocate budget for.
        total_budget : int
            Total token budget.
        hint : str or None
            Query hint for prioritizing columns.
        include_schema : bool
            Whether schema section is enabled.
        include_stats : bool
            Whether stats section is enabled.
        include_samples : bool
            Whether samples section is enabled.

        Returns
        -------
        BudgetPlan

        """
        # Compute active ratios
        active: dict[str, float] = {}
        if include_schema:
            active["schema"] = self._ratio.get("schema", 0.15)
        if include_stats:
            active["stats"] = self._ratio.get("stats", 0.55)
        if include_samples:
            active["samples"] = self._ratio.get("samples", 0.30)

        if not active:
            return BudgetPlan(
                schema_budget=0, total_budget=total_budget
            )

        # Normalize ratios
        ratio_sum = sum(active.values())
        normalized = {k: v / ratio_sum for k, v in active.items()}

        schema_budget = int(
            total_budget * normalized.get("schema", 0)
        )
        stats_budget = int(
            total_budget * normalized.get("stats", 0)
        )
        sample_budget = int(
            total_budget * normalized.get("samples", 0)
        )

        # Distribute stats budget across columns
        columns = list(df.columns)
        column_budgets = _distribute_column_budget(
            df, columns, stats_budget, hint
        )

        return BudgetPlan(
            schema_budget=schema_budget,
            column_budgets=column_budgets,
            sample_budget=sample_budget,
            total_budget=total_budget,
        )


def _distribute_column_budget(
    df: pd.DataFrame,
    columns: list[str],
    stats_budget: int,
    hint: str | None,
) -> dict[str, int]:
    """Distribute stats budget across columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame.
    columns : list[str]
        Column names to distribute budget across.
    stats_budget : int
        Total stats budget to distribute.
    hint : str or None
        Query hint for boosting relevant columns.

    Returns
    -------
    dict[str, int]
        Budget per column.

    """
    if not columns:
        return {}

    # Base weight: equal distribution
    weights: dict[str, float] = {col: 1.0 for col in columns}

    # Boost by cardinality (more unique values = more interesting)
    for col in columns:
        nunique = df[col].nunique()
        n = len(df)
        if n > 0:
            ratio = nunique / n
            # High cardinality gets a small boost
            weights[col] *= 1.0 + ratio * 0.5

    # Reduce weight for high-null columns
    for col in columns:
        null_rate = float(df[col].isna().mean())
        weights[col] *= 1.0 - null_rate * 0.5

    # Apply hint boosting
    if hint:
        for col in columns:
            sample_vals: list[Any] = []
            with contextlib.suppress(Exception):
                sample_vals = df[col].dropna().head(5).tolist()
            relevance = compute_hint_relevance(
                hint, col, sample_vals
            )
            if relevance > 0:
                weights[col] *= 1.0 + relevance * 2.0

    # Normalize and allocate
    total_weight = sum(weights.values())
    if total_weight == 0:
        per_col = stats_budget // max(len(columns), 1)
        return {col: per_col for col in columns}

    result: dict[str, int] = {}
    for col in columns:
        result[col] = max(
            10, int(stats_budget * weights[col] / total_weight)
        )

    return result
