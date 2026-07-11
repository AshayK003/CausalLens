# Contributing to CausalLens

Thanks for your interest in contributing. This document covers the practical details.

## Getting Started

```bash
git clone https://github.com/AshayK003/CausalLens.git
cd CausalLens
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -e ".[dev]"
```

Verify everything works:

```bash
pytest tests/ -v             # all tests pass
streamlit run app.py         # dashboard launches
```

## Development Workflow

1. Create a branch from `master`: `git checkout -b feature/my-change`
2. Make your changes
3. Add or update tests
4. Run `pytest tests/ -v` — all tests must pass
5. Commit and push
6. Open a pull request

## What to Work On

Check [open issues](https://github.com/AshayK003/CausalLens/issues) for planned work. Good first contributions:

- Add unit tests for app.py helpers (see issue #1)
- Add tests for subgroup analysis (see issue #2)
- Fix PDF report label overlap (see issue #3)
- Fix metric card delta color (see issue #4)

## Code Conventions

### Python

- Follow existing style in the file you're editing
- Use `logging` module for debug/info output
- Keep imports sorted: stdlib, third-party, local
- Type hints are encouraged but not required

### Architecture

- `src/core/` contains analysis methods (ARIMA, BSTS, DiD, Synthetic Control)
- `src/data/` handles dataset loading and preprocessing
- `src/reports/` generates PDF and HTML exports
- `app.py` is the Streamlit UI

### Testing

- Test files go in `tests/`
- Name tests `test_<module>.py`
- Use `pytest` fixtures
- Write behavior-focused tests, not implementation tests

## Commit Messages

Use short imperative descriptions:

```
add test coverage for subgroup analysis
fix PDF report x-axis label overlap
update ARIMA order selection logic
improve error messages for invalid dates
```

## Pull Requests

- Keep PRs focused — one change per PR
- Include a description of what changed and why
- Reference related issues

## Questions?

Open an issue or start a discussion on GitHub.
