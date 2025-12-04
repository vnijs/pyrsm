# Makefile for pyrsm package testing and deployment
# Usage: make <target>

.PHONY: help test test-basics test-model test-eda test-all notebooks notebooks-basics notebooks-model notebooks-eda lint format clean build publish publish-test

# Notebook runner - executes and fails on error
NBCONVERT = uv run jupyter nbconvert --to notebook --execute --inplace

# Default target
help:
	@echo "pyrsm - Testing and Deployment Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Run all pytest tests"
	@echo "  make test-basics     - Run basics module tests only"
	@echo "  make test-model      - Run model module tests only"
	@echo "  make test-eda        - Run eda module tests only"
	@echo "  make test-verbose    - Run all tests with verbose output"
	@echo ""
	@echo "Notebooks:"
	@echo "  make notebooks       - Execute all example notebooks"
	@echo "  make notebooks-basics - Execute basics notebooks only"
	@echo "  make notebooks-model - Execute model notebooks only"
	@echo "  make notebooks-eda   - Execute eda notebooks only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            - Run ruff linter"
	@echo "  make format          - Format code with black"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  make clean           - Remove build artifacts"
	@echo "  make build           - Build package"
	@echo "  make publish-test    - Publish to TestPyPI"
	@echo "  make publish         - Publish to PyPI"
	@echo ""
	@echo "Combined:"
	@echo "  make test-all        - Run all tests and notebooks"
	@echo "  make check           - Run lint, tests, and notebooks"

# =============================================================================
# Testing
# =============================================================================

test:
	uv run pytest tests/ -v --tb=short

test-basics:
	uv run pytest tests/test_basics.py tests/test_single_*.py tests/test_compare_*.py tests/test_correlation*.py tests/test_goodness.py tests/test_cross_tabs.py tests/test_central_limit_theorem.py -v --tb=short

test-model:
	uv run pytest tests/test_regression.py tests/test_logistic.py tests/test_perf.py tests/test_mlp.py tests/test_rforest.py tests/test_xgb.py tests/test_model_utils.py -v --tb=short

test-eda:
	uv run pytest tests/test_eda.py -v --tb=short

test-verbose:
	uv run pytest tests/ -v --tb=long

test-coverage:
	uv run pytest tests/ -v --cov=pyrsm --cov-report=term-missing

# =============================================================================
# Notebook Execution
# Each notebook command will fail immediately if execution fails
# =============================================================================

notebooks-basics:
	@echo "=== Executing basics notebooks ==="
	$(NBCONVERT) examples/basics/basics-compare-means.ipynb
	$(NBCONVERT) examples/basics/basics-compare-props.ipynb
	$(NBCONVERT) examples/basics/basics-correlation.ipynb
	$(NBCONVERT) examples/basics/basics-cross-tabs.ipynb
	$(NBCONVERT) examples/basics/basics-goodness.ipynb
	$(NBCONVERT) examples/basics/basics-probability-calculator.ipynb
	$(NBCONVERT) examples/basics/basics-single-proportion.ipynb
	@echo "=== All basics notebooks passed ==="

notebooks-model:
	@echo "=== Executing model notebooks ==="
	$(NBCONVERT) examples/model/model-linear-regression.ipynb
	$(NBCONVERT) examples/model/model-logistic-regression.ipynb
	$(NBCONVERT) examples/model/model-mlp-classification.ipynb
	$(NBCONVERT) examples/model/model-mlp-regression.ipynb
	$(NBCONVERT) examples/model/model-rforest-classification.ipynb
	$(NBCONVERT) examples/model/model-rforest-regression.ipynb
	$(NBCONVERT) examples/model/model-xgboost-classification.ipynb
	$(NBCONVERT) examples/model/model-xgboost-regression.ipynb
	@echo "=== All model notebooks passed ==="

notebooks-eda:
	@echo "=== Executing eda notebooks ==="
	$(NBCONVERT) examples/eda/eda-explore.ipynb
	$(NBCONVERT) examples/eda/eda-pivot.ipynb
	$(NBCONVERT) examples/eda/eda-visualize.ipynb
	@echo "=== All eda notebooks passed ==="

notebooks-data:
	@echo "=== Executing data notebooks ==="
	$(NBCONVERT) examples/data/load-example-data.ipynb
	$(NBCONVERT) examples/data/save-load-state.ipynb
	@echo "=== All data notebooks passed ==="

notebooks: notebooks-basics notebooks-model notebooks-eda notebooks-data
	@echo "=== All notebooks executed successfully ==="

# =============================================================================
# Code Quality
# =============================================================================

lint:
	uv run ruff check pyrsm/ tests/

lint-fix:
	uv run ruff check pyrsm/ tests/ --fix

format:
	uv run black pyrsm/ tests/

format-check:
	uv run black pyrsm/ tests/ --check

# =============================================================================
# Build & Deploy
# =============================================================================

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

publish-test: build
	@echo "Publishing to TestPyPI..."
	uv publish --publish-url https://test.pypi.org/legacy/
	@echo ""
	@echo "Install from TestPyPI with:"
	@echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyrsm"

publish: build
	@echo "Publishing to PyPI..."
	@read -p "Are you sure you want to publish to PyPI? [y/N] " confirm && [ "$$confirm" = "y" ]
	uv publish
	@echo "Published to PyPI successfully!"

# =============================================================================
# Combined Targets
# =============================================================================

test-all: test notebooks
	@echo "=== All tests and notebooks passed ==="

check: lint test notebooks
	@echo "=== All checks passed ==="

# Quick smoke test (fast subset)
smoke:
	uv run pytest tests/test_basics.py tests/test_stats.py -v --tb=short -x
