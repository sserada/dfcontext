# dfcontext

> Generate optimal LLM context from pandas DataFrames within a token budget.

[![PyPI version](https://badge.fury.io/py/dfcontext.svg)](https://pypi.org/project/dfcontext/)
[![Python](https://img.shields.io/pypi/pyversions/dfcontext)](https://pypi.org/project/dfcontext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/sserada/dfcontext/actions/workflows/ci.yml/badge.svg)](https://github.com/sserada/dfcontext/actions)

## Install

```bash
pip install dfcontext
```

## Quick Start

```python
import pandas as pd
from dfcontext import to_context

df = pd.read_csv("sales.csv")  # 100K rows
ctx = to_context(df, token_budget=2000)
print(ctx)
```

## License

MIT
