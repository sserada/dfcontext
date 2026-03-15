"""Use dfcontext with OpenAI GPT.

Prerequisites:
    pip install dfcontext[all] openai

Usage:
    export OPENAI_API_KEY="your-key"
    python with_openai.py
"""

import numpy as np
import pandas as pd
from openai import OpenAI

from dfcontext import to_context

# --- Prepare data ---
np.random.seed(42)
df = pd.DataFrame({
    "customer_id": range(10_000),
    "age": np.random.normal(35, 12, 10_000).clip(18, 80).astype(int),
    "plan": np.random.choice(["free", "basic", "premium"], 10_000, p=[0.6, 0.25, 0.15]),
    "monthly_spend": np.random.exponential(50, 10_000).round(2),
    "days_active": np.random.randint(1, 365, 10_000),
    "churned": np.random.choice([True, False], 10_000, p=[0.15, 0.85]),
})

# --- Generate context with a churn-focused hint ---
ctx = to_context(df, token_budget=1500, hint="churn prediction factors")

print("=== Context ===")
print(ctx)
print()

# --- Send to GPT ---
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": (
            f"{ctx}\n\n"
            "Based on this data overview, what features seem most "
            "relevant for predicting customer churn? Suggest a modeling approach."
        ),
    }],
)

print("=== GPT's Analysis ===")
print(response.choices[0].message.content)
