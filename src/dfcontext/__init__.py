"""dfcontext — Generate optimal LLM context from pandas DataFrames."""

from dfcontext._version import __version__
from dfcontext.analyzers.base import ColumnSummary
from dfcontext.budget import BudgetPlan
from dfcontext.config import ContextConfig
from dfcontext.core import analyze_columns, count_tokens, to_context

__all__ = [
    "BudgetPlan",
    "ColumnSummary",
    "ContextConfig",
    "__version__",
    "analyze_columns",
    "count_tokens",
    "to_context",
]
