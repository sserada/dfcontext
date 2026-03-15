"""See how different token budgets affect dfcontext output.

No API key required — this example runs locally and shows you
what information is included at each budget level.

Prerequisites:
    pip install dfcontext[all]

Usage:
    python budget_tuning.py
"""

import numpy as np
import pandas as pd

from dfcontext import ContextConfig, count_tokens, to_context

# --- Sample data ---
np.random.seed(42)
n = 10_000

df = pd.DataFrame({
    "region": np.random.choice(["East", "West", "North", "South", "Central"], n),
    "product": np.random.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y"], n),
    "sales": np.random.exponential(1000, n).round(2),
    "quantity": np.random.randint(1, 200, n),
    "date": pd.date_range("2023-01-01", periods=n, freq="h"),
    "is_return": np.random.choice([True, False], n, p=[0.07, 0.93]),
})

# --- Compare budgets ---
budgets = [200, 500, 1000, 2000, 5000]

for budget in budgets:
    ctx = to_context(df, token_budget=budget)
    actual_tokens = count_tokens(ctx)
    lines = ctx.count("\n") + 1
    has_stats = "Column statistics" in ctx
    has_samples = "Sample rows" in ctx

    print(f"Budget: {budget:>5} tokens")
    print(f"  Actual:  {actual_tokens:>5} tokens, {lines:>3} lines")
    print(f"  Stats:   {'yes' if has_stats else 'no'}")
    print(f"  Samples: {'yes' if has_samples else 'no'}")
    print()

# --- Compare sections ---
print("=" * 60)
print("Schema only vs Full context")
print("=" * 60)
print()

schema_only = to_context(
    df,
    token_budget=2000,
    include_stats=False,
    include_samples=False,
)
print(f"Schema only: {count_tokens(schema_only)} tokens")
print(schema_only)
print()

stats_only = to_context(
    df,
    token_budget=2000,
    include_schema=False,
    include_samples=False,
)
print(f"Stats only: {count_tokens(stats_only)} tokens")
print(stats_only)
print()

# --- Custom budget ratios ---
print("=" * 60)
print("Custom budget ratio: more stats, fewer samples")
print("=" * 60)
print()

config = ContextConfig(
    token_budget=1500,
    budget_ratio={"schema": 0.10, "stats": 0.75, "samples": 0.15},
)
ctx = to_context(df, config=config)
print(f"Custom ratio: {count_tokens(ctx)} tokens")
print(ctx)
