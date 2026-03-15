# Contributing to dfcontext

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/sserada/dfcontext.git
cd dfcontext

# Install dependencies (requires uv)
uv sync --all-extras

# Verify everything works
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest
```

## Workflow

1. **Check existing issues** or create a new one describing the change
2. **Fork & create a branch** from `main` (e.g. `feat/new-feature`, `fix/bug-name`)
3. **Make changes** with tests
4. **Run checks** before pushing:

```bash
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run mypy src/                # Type check
uv run pytest --cov=dfcontext   # Test with coverage
```

5. **Open a Pull Request** against `main`

## Coding Conventions

- **Python 3.10+** syntax (`X | Y` union types)
- **Type hints** on all public functions
- **NumPy-style docstrings** on all public functions
- **~300 lines max** per file
- **All code, comments, issues, and PRs in English**

## Testing

- Write tests for all new functionality
- Target **90%+** coverage
- Test edge cases: empty DataFrames, all-null columns, single-value columns

## Design Principles

These are non-negotiable:

1. **No LLM calls** — dfcontext is a pure data processing library
2. **Minimal dependencies** — only pandas + numpy are required
3. **Output is always `str`** — works with any LLM provider
4. **Token budget is strict** — output must never exceed `token_budget`
5. **Column-type-aware** — each column type gets its own analysis strategy

## Branch Naming

- `feat/<description>` — new features
- `fix/<description>` — bug fixes
- `refactor/<description>` — code refactoring
- `docs/<description>` — documentation
- `ci/<description>` — CI/CD changes
- `chore/<description>` — maintenance

## Reporting Bugs

Use the [bug report template](https://github.com/sserada/dfcontext/issues/new?template=bug_report.yml) and include:

- Minimal reproduction code
- Expected vs actual behavior
- dfcontext and Python versions

## Questions

For questions and discussions, use [GitHub Discussions](https://github.com/sserada/dfcontext/discussions).
