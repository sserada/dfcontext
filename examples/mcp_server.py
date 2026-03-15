"""Build an MCP server tool that summarizes datasets with dfcontext.

This creates a tool that any MCP client (Claude Desktop, etc.) can call
to get an intelligent summary of a CSV file.

Prerequisites:
    pip install dfcontext[all] fastmcp

Usage:
    python mcp_server.py
    # Then connect via an MCP client
"""

import pandas as pd
from fastmcp import FastMCP

from dfcontext import to_context

mcp = FastMCP("DataAnalyzer")


@mcp.tool()
def describe_dataset(
    file_path: str,
    question: str = "",
    token_budget: int = 2000,
) -> str:
    """Summarize a CSV dataset for LLM analysis.

    Args:
        file_path: Path to a CSV file.
        question: Optional question to focus the summary on relevant columns.
        token_budget: Maximum tokens for the summary (default 2000).

    Returns:
        A structured summary of the dataset.
    """
    df = pd.read_csv(file_path)

    ctx = to_context(
        df,
        token_budget=token_budget,
        hint=question if question else None,
    )

    if question:
        return f"{ctx}\n\n(Focus: {question})"
    return ctx


@mcp.tool()
def compare_datasets(
    file_path_a: str,
    file_path_b: str,
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
    token_budget: int = 3000,
) -> str:
    """Compare two CSV datasets side by side.

    Args:
        file_path_a: Path to the first CSV file.
        file_path_b: Path to the second CSV file.
        label_a: Label for the first dataset.
        label_b: Label for the second dataset.
        token_budget: Total token budget (split between both datasets).

    Returns:
        A side-by-side summary of both datasets.
    """
    per_dataset = token_budget // 2

    df_a = pd.read_csv(file_path_a)
    df_b = pd.read_csv(file_path_b)

    ctx_a = to_context(df_a, token_budget=per_dataset)
    ctx_b = to_context(df_b, token_budget=per_dataset)

    return f"## {label_a}\n{ctx_a}\n\n## {label_b}\n{ctx_b}"


if __name__ == "__main__":
    mcp.run()
