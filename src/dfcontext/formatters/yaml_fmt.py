"""YAML output formatter (requires pyyaml)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfcontext.formatters.base import BaseFormatter

if TYPE_CHECKING:
    import pandas as pd

    from dfcontext.analyzers.base import ColumnSummary


class YAMLFormatter(BaseFormatter):
    """Format output as YAML.

    Requires the ``pyyaml`` optional dependency.
    """

    def format_schema(self, df: pd.DataFrame) -> str:
        """Format schema as YAML."""
        yaml = _import_yaml()

        schema: dict[str, Any] = {
            "dataset": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
            },
            "schema": {},
        }
        for col in df.columns:
            non_null_pct = round(df[col].notna().mean() * 100)
            schema["schema"][str(col)] = {
                "type": str(df[col].dtype),
                "non_null_pct": non_null_pct,
            }
        return str(
            yaml.dump(
                schema,
                default_flow_style=False,
                allow_unicode=True,
            )
        ).rstrip()

    def format_stats(
        self, summaries: list[ColumnSummary]
    ) -> str:
        """Format column statistics as YAML."""
        yaml = _import_yaml()

        stats_dict: dict[str, Any] = {}
        for s in summaries:
            entry: dict[str, Any] = {
                "type": s.column_type,
                "non_null_rate": round(s.non_null_rate, 4),
            }
            entry.update(s.stats)
            if s.distribution_sketch:
                entry["distribution"] = s.distribution_sketch
            stats_dict[s.name] = entry

        return str(
            yaml.dump(
                {"statistics": stats_dict},
                default_flow_style=False,
                allow_unicode=True,
            )
        ).rstrip()

    def format_samples(
        self, df: pd.DataFrame, max_rows: int
    ) -> str:
        """Format sample rows as YAML."""
        yaml = _import_yaml()

        if df.empty or max_rows <= 0:
            return ""

        sample = df.head(max_rows)
        rows: list[dict[str, Any]] = []
        for _, row in sample.iterrows():
            rows.append(
                {str(c): _convert(row[c]) for c in sample.columns}
            )

        return str(
            yaml.dump(
                {"samples": rows},
                default_flow_style=False,
                allow_unicode=True,
            )
        ).rstrip()


def _import_yaml() -> Any:
    """Import yaml, raising ImportError with helpful message."""
    try:
        import yaml

        return yaml
    except ImportError:
        msg = (
            "YAML format requires pyyaml. "
            "Install it with: pip install dfcontext[yaml]"
        )
        raise ImportError(msg) from None


def _convert(val: Any) -> Any:
    """Convert pandas/numpy types to Python native for YAML."""
    import numpy as np

    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if hasattr(val, "isoformat"):
        return str(val)
    return val
