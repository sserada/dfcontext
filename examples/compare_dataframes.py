"""Compare two DataFrames side by side with dfcontext.

A common use case: year-over-year comparison, A/B test results,
or before/after analysis. Generate context for each DataFrame
and let the LLM find the differences.

Prerequisites:
    pip install dfcontext[all] anthropic

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python compare_dataframes.py
"""

import anthropic
import numpy as np
import pandas as pd

from dfcontext import count_tokens, to_context

# --- Simulate 2023 vs 2024 sales data ---
np.random.seed(42)

df_2023 = pd.DataFrame({
    "region": np.random.choice(["East", "West", "North", "South"], 20_000),
    "sales": np.random.exponential(800, 20_000).round(2),
    "quantity": np.random.randint(1, 80, 20_000),
    "date": pd.date_range("2023-01-01", periods=20_000, freq="h"),
})

df_2024 = pd.DataFrame({
    "region": np.random.choice(["East", "West", "North", "South"], 25_000),
    "sales": np.random.exponential(1200, 25_000).round(2),  # Growth!
    "quantity": np.random.randint(1, 120, 25_000),
    "date": pd.date_range("2024-01-01", periods=25_000, freq="h"),
})

# --- Generate context for each (split the budget) ---
# Use the same hint so both focus on comparable aspects
hint = "sales volume and regional distribution"

ctx_2023 = to_context(df_2023, token_budget=800, hint=hint)
ctx_2024 = to_context(df_2024, token_budget=800, hint=hint)

# --- Combine into a comparison prompt ---
prompt = f"""## 2023 Data
{ctx_2023}

## 2024 Data
{ctx_2024}

Compare the two years. What changed? Identify growth areas and concerns."""

print("=== Combined prompt length ===")
print(f"{count_tokens(prompt)} tokens")
print()

# --- Send to Claude ---
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
)

print("=== Year-over-Year Analysis ===")
print(response.content[0].text)
