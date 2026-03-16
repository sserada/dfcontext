# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-03-16

### Added

- Column correlation detection with `find_top_correlations()` and `include_correlations` parameter
- Outlier detection via IQR method for numeric columns (`outlier_rate` stat)
- Adaptive budget allocation with tiered statistics (P5/P95, skewness only for budget > 200)
- `exclude_columns` parameter for `to_context()` to omit sensitive columns

### Fixed

- Non-string column names (int, tuple) no longer cause KeyError in budget allocation
- Dynamic min-per-column budget prevents allocation overflow with many columns

### Changed

- Statistics detail now scales with token budget (Tier 1/2/3)
- Extracted shared `StatsBlock` for DRY formatter logic
- Expanded ruff lint rules: D (docstrings), PT (pytest), RUF (ruff-specific)
- Standardized pandas import pattern across codebase

## [0.1.1] - 2026-03-16

### Fixed

- Sampler crash with non-integer DataFrame indices (string, datetime, etc.)
- Incorrect granularity label ("sub-second") for identical datetime values
- Markdown table corruption when column names or values contain pipe (`|`) characters

### Added

- LLM integration examples: Claude, OpenAI, MCP server, budget tuning, DataFrame comparison

## [0.1.0] - 2026-03-15

### Added

- `to_context()` main API for generating LLM context from DataFrames
- `analyze_columns()` for structured column analysis
- `count_tokens()` for token counting
- `ContextConfig` dataclass for advanced configuration
- Column-type-aware analyzers: numeric (with mini histogram), categorical, text (with pattern detection), datetime (with granularity estimation), boolean
- Token budget management with `TokenBudgetAllocator`
- Query hints for prioritizing relevant columns
- Output formatters: Markdown, plain text, YAML
- Representative row sampling (stratified and diversity-based)
- tiktoken integration with character-based fallback
- GitHub Actions CI (Python 3.10–3.13)
- PyPI publishing via trusted publishing

[0.2.0]: https://github.com/sserada/dfcontext/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/sserada/dfcontext/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sserada/dfcontext/releases/tag/v0.1.0
