# Repository Guidelines

## Project Structure & Modules
- `pyrsm/` contains core analytics code: `basics/` and `bins.py` for data prep, `design/`, `eda/`, `model/`, `multivariate/`, `stats.py`, and utilities in `utils.py`, plus `example_data.py` for sample datasets.
- App-facing assets live in `pyrsm/shiny-app`, `pyrsm/radiant`, and `streamlet/`; keep shared images in `images/`.
- Tests sit under `tests/` (mirrors module names), with plotting snapshots in `tests/plot_comparisons/`.
- Reference `PLOTTING_MIGRATION.md` when touching plotting logic; sample workflows belong in `examples/`.

## Setup, Build, and Local Dev
- Use Python 3.12+. Recommended flow with uv:
  - `uv venv --python 3.12 && source .venv/bin/activate`
  - `uv sync` to install locked deps (see `uv.lock`).
  - `uv run python -m pip install -e .` if you need an editable install.
- Package build: `uv build` (generates sdist/wheel under `dist/`).
- Quick sanity check: `uv run python -c "import pyrsm; print(pyrsm.__version__)"`.

## Test Workflow
- Run all tests with `uv run pytest`; target `tests/` automatically via `pytest.ini`.
- Scope runs: `uv run pytest tests/test_bins.py -k some_case`.
- Add regression tests for bug fixes; name files `test_<module>.py` and functions `test_<behavior>` to keep discovery consistent.

## Coding Style & Formatting
- Follow Ruff + Black: line length 100, Python 3.12 targeting. Lint via `uv run ruff check .`; format via `uv run black pyrsm tests`.
- Avoid bare `except`; prefer explicit exceptions (Ruff ignores `E722` only where necessary).
- Use snake_case for functions/vars/modules, CapWords for classes, and keep public APIs typed where possible.
- Keep imports ordered (Ruff `I` rules) and prefer pure functions in analytics helpers to ease testing.

## Commit & Pull Request Guidelines
- Git history favors short, lower-case subjects (present tense). Example: `update eda bins` or `fix regression stats`.
- Include a brief body when context is non-obvious (what/why, risk, follow-ups). Reference issues where applicable.
- PRs should note:
  - Summary of change and affected modules.
  - Tests run (`uv run pytest`, specific files) and any data/plot artifacts touched.
  - Screenshots/GIFs for UI-facing changes in `shiny-app` or `streamlet`.
  - Dependency or data additions (location and rationale).
