# dfcontext Examples

Practical examples showing how to use dfcontext with LLMs and data workflows.

| Example | Description |
|---------|-------------|
| [with_claude.py](with_claude.py) | Analyze a DataFrame with Anthropic Claude |
| [with_openai.py](with_openai.py) | Analyze a DataFrame with OpenAI GPT |
| [compare_dataframes.py](compare_dataframes.py) | Compare two DataFrames (e.g. year-over-year) |
| [budget_tuning.py](budget_tuning.py) | See how different token budgets affect output |
| [mcp_server.py](mcp_server.py) | Build an MCP server tool that summarizes datasets |

## Prerequisites

```bash
pip install dfcontext[all]

# For LLM examples:
pip install anthropic   # with_claude.py
pip install openai      # with_openai.py
pip install fastmcp     # mcp_server.py
```
