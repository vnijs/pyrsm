# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Run all tests: `python -m unittest discover -s tests`
- Run single test: `python -m tests.test_file_name` (e.g., `python -m tests.test_utils`)
- Lint code with ruff: `ruff check pyrsm/`
- Format code with black: `black pyrsm/`
- Check dependencies: `python check_versions.py`

## Package Management
- Use UV for package management: `uv add <package>` instead of pip install
- For development: `uv pip install -e .` for editable installs
- Dependencies are tracked in both pyproject.toml and uv.lock

## Code Style Guidelines
- Line length: 100 characters max (as per ruff config)
- Use black for formatting
- Use ruff for linting (E, F, I, UP rules enabled)
- Function/variable names: snake_case
- Class names: PascalCase
- Import style: Use absolute imports; sort imports with ruff
- Error handling: Allows bare excepts (E722 ignored in ruff)
- Python target: 3.12+
- In __init__.py files: star imports allowed (F403, F405 ignored in ruff)
- Doc style: Use functions/class docstrings for documentation