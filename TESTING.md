# Testing Guide for pyrsm

This document describes how to run tests and validate the pyrsm package.

## Prerequisites

Ensure you have UV installed and the virtual environment set up:

```bash
source .venv/bin/activate
uv sync
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `make test` | Run all pytest tests |
| `make notebooks` | Execute all example notebooks |
| `make test-all` | Run tests + notebooks |
| `make check` | Run lint + tests + notebooks |

## Running Tests

### All Tests

```bash
make test
```

Or directly with pytest:

```bash
uv run pytest tests/ -v --tb=short
```

### Module-Specific Tests

**Basics module:**
```bash
make test-basics
```

**Model module:**
```bash
make test-model
```

**EDA module:**
```bash
make test-eda
```

### Verbose Output

```bash
make test-verbose
```

### Test Coverage

```bash
make test-coverage
```

## Running Example Notebooks

Notebooks are executed in-place to verify they run without errors.

### All Notebooks

```bash
make notebooks
```

### By Category

```bash
make notebooks-basics   # basics module examples
make notebooks-model    # model module examples
make notebooks-eda      # eda module examples
make notebooks-data     # data loading examples
```

### Single Notebook

```bash
uv run jupyter nbconvert --to notebook --execute --inplace examples/basics/basics-compare-means.ipynb
```

## Code Quality

### Linting

```bash
make lint          # Check for issues
make lint-fix      # Auto-fix issues
```

### Formatting

```bash
make format        # Format code
make format-check  # Check formatting without changes
```

## Building and Publishing

### Clean Build Artifacts

```bash
make clean
```

### Build Package

```bash
make build
```

### Publish to TestPyPI

```bash
make publish-test
```

Then test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyrsm
```

### Publish to PyPI

```bash
make publish
```

## Pre-Release Checklist

Before releasing a new version:

1. **Run all tests:**
   ```bash
   make test
   ```

2. **Execute all notebooks:**
   ```bash
   make notebooks
   ```

3. **Check code quality:**
   ```bash
   make lint
   make format-check
   ```

4. **Full validation:**
   ```bash
   make check
   ```

5. **Test on TestPyPI first:**
   ```bash
   make publish-test
   ```

6. **Publish to PyPI:**
   ```bash
   make publish
   ```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_basics.py           # Basic functionality tests
├── test_compare_means.py    # compare_means module
├── test_compare_props.py    # compare_props module
├── test_correlation_ext.py  # correlation module
├── test_cross_tabs.py       # cross_tabs module
├── test_goodness.py         # goodness of fit tests
├── test_single_prop.py      # single_prop module
├── test_single_mean.py      # single_mean module
├── test_central_limit_theorem.py
├── test_regression.py       # linear regression
├── test_logistic.py         # logistic regression
├── test_mlp.py              # MLP models
├── test_rforest.py          # Random forest
├── test_xgb.py              # XGBoost
├── test_perf.py             # Model performance
├── test_eda.py              # EDA module
├── test_stats.py            # Statistics utilities
├── test_utils.py            # General utilities
└── plot_comparisons/        # Visual regression baselines
```

## Troubleshooting

### Notebook execution fails

If a notebook fails, run it manually to see the error:

```bash
uv run jupyter notebook examples/basics/basics-compare-means.ipynb
```

### Import errors

Ensure the package is installed in development mode:

```bash
uv pip install -e .
```

### Test discovery issues

Check pytest configuration in `pytest.ini` or `pyproject.toml`.
