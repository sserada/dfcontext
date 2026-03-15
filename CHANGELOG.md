# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[0.1.1]: https://github.com/sserada/dfcontext/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sserada/dfcontext/releases/tag/v0.1.0
