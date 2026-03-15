"""Use dfcontext with Anthropic Claude.

This example shows how to generate DataFrame context and pass it
to Claude for analysis. The key idea: instead of dumping raw data,
dfcontext gives Claude a compact but statistically rich summary.

Prerequisites:
    pip install dfcontext[all] anthropic

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python with_claude.py
"""

import anthropic
import numpy as np
import pandas as pd

from dfcontext import to_context

# --- 1. Prepare sample data ---
np.random.seed(42)
n = 50_000

df = pd.DataFrame({
    "region": np.random.choice(["East", "West", "North", "South"], n),
    "product": np.random.choice(["Widget A", "Widget B", "Gadget X"], n),
    "sales": np.random.exponential(1000, n).round(2),
    "quantity": np.random.randint(1, 100, n),
    "date": pd.date_range("2023-01-01", periods=n, freq="h"),
    "is_return": np.random.choice([True, False], n, p=[0.06, 0.94]),
})

# --- 2. Generate context ---
# Without dfcontext, df.to_string() would produce ~2M tokens.
# With dfcontext, we get a rich summary in ~600 tokens.
ctx = to_context(df, token_budget=1000, hint="regional sales performance")

print("=== Generated Context ===")
print(ctx)
print()

# --- 3. Send to Claude ---
client = anthropic.Anthropic()

question = (
    "What are the key insights about regional sales "
    "performance? Are there any notable patterns?"
)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"{ctx}\n\n{question}",
    }],
)

print("=== Claude's Analysis ===")
print(response.content[0].text)


# --- Tips ---
# 1. Use `hint` to focus on what you're analyzing:
#    ctx = to_context(df, hint="return rate by product")
#
# 2. Adjust budget based on your needs:
#    - 500 tokens: quick overview (schema + basic stats)
#    - 2000 tokens: standard analysis (full stats + samples)
#    - 5000 tokens: detailed analysis (everything + more samples)
#
# 3. For multi-turn conversations, generate context once and reuse:
#    ctx = to_context(df, token_budget=2000)
#    messages = [{"role": "user", "content": f"{ctx}\n\nFirst question..."}]
#    # ... get response, then ask follow-ups without resending ctx
