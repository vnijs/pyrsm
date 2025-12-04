# Polars + Plotnine Transition Plan (Basics First)

## Scope & Goals
- Freeze current `pyrsm.basics.compare_means` behavior (pandas + polars inputs) with comprehensive pytest coverage and plot artifacts.
- Improve docs so LLMs/MCP can discover usage patterns; leverage existing basics datasets and example notebooks.
- Establish repeatable baselines before switching default DF to polars and migrating plotting to plotnine.

## Baseline Artifacts to Produce
- Tests: granular pytest modules for `compare_means` (replacing the ad-hoc coverage in `tests/test_basics.py`).
- Fixtures: dataset loaders for parquet basics data; synthetic edge-case frames; temporary plot output directory.
- Plot snapshots: saved matplotlib outputs in `tests/plot_comparisons/basics/compare_means/` (one per plot type).
- Docs: enriched `compare_means` docstring + dataset references to help LLMs/MCP; note notebook path `examples/basics/basics-compare-means.ipynb`.

## Test Design (compare_means)
- Parametrize pandas/polars inputs; honor `check_dataframe` behavior.
- Data cases: basics parquet datasets (`salary`, `consider`, `demand_uk`, `newspaper`) plus synthetic sets with nulls, zero-variance, unequal group sizes.
- Functional checks:
  - Numeric `var1` melt path when `var2` is list; category casting and level tracking.
  - Independent vs paired (size mismatch error), `t-test` vs `wilcox`, alt hypotheses, CI label formatting.
  - `comb` handling and `adjust="bonferroni"` p-value adjustments; sig star mapping.
  - Descriptive stats integrity (`mean`, `n`, `n_missing`, `sd`, `se`, `me`); handling of missing values.
  - `summary(extra=False/True)` structure (column presence, not exact strings); attribute types (`descriptive_stats`, `comp_stats`, `levels`, `name` from dict input).
- Plots: ensure `plot` returns a figure/axes and writes non-empty PNGs for `scatter`, `box`, `density`, `bar` under `tests/plot_comparisons/basics/compare_means/`.

## Documentation Enhancements
- Expand `compare_means` docstring with pandas/polars quick-start using `load_data(pkg="basics", name="salary")`, parameter notes, return attributes, and expected outputs.
- Add brief dataset pointers (e.g., in `pyrsm/data/basics/__init__.py` or module docs) linking to `*_description.md` for LLM discoverability.
- Keep concise comments explaining non-obvious logic (melt path, Welch DOF helper).

## Execution Steps
1) Add fixtures in `tests/conftest.py` for basics parquet loaders (pandas/polars) and temp plot dir.
2) Create `tests/test_compare_means.py` with parametrized cases above; remove/retire `compare_means` stubs from `tests/test_basics.py`.
3) Generate plot PNGs during tests and assert file presence/size; commit artifacts.
4) Run `uv run pytest` to set baseline; document any current failures/quirks as known issues.
5) Update docs per above; ensure references to example notebook for further context.
6) Summarize baseline coverage + gaps to guide subsequent polars-default and plotnine migration.

## Risks / Notes
- Some breaking changes acceptable later for clarity, but baseline should reflect current behavior now.
- Plotnine migration will alter plot outputs; current PNGs act as comparison for the changeover.

## Output Formatting Generalization (basics/)
- Goal: adopt the `compare_means` dual-output pattern (rich/styled in notebooks or VS Code, plain text in terminals) across all basics tools.
- Discovery:
  - Catalogue which basics modules produce summaries/printed output (`cross_tabs`, `correlation`, `compare_props`, `single_mean`, `single_prop`, `goodness`, `probability_calculator*`, `central_limit_theorem`, plotting helpers).
  - Note current output types (pandas DataFrame, polars DataFrame, plain `print`, matplotlib figures) and existing styles.
- Design shared utilities:
  - Add a formatter helper (e.g., `pyrsm.basics.display_utils`) that:
    - Detects environment (Jupyter/IPython, VS Code/Notebook renderer, or plain terminal).
    - Provides `render_tables(desc_df, comp_df, extra, dec)` that routes to styled (pandas/gt/Styler) vs plain (polars Config print) outputs.
    - Exposes a small header printer helper to standardize metadata blocks.
  - Keep dependencies minimal; reuse existing `great_tables` usage if available, otherwise fall back to pandas Styler.
- Integration plan:
  - Refactor `compare_means` to consume the shared helper (maintaining current behavior) as the reference implementation.
  - Incrementally update other basics modules to use the helper; ensure they surface their summary data as pandas/polars DataFrames compatible with the renderer.
  - For modules without structured outputs, add lightweight summary DataFrames to feed the renderer where appropriate.
- Testing:
  - Add unit tests for the helper to confirm environment detection fallbacks (mock IPython presence) and plain output formatting doesn’t raise.
  - Add smoke tests per module ensuring `summary()` executes in “plain” mode and returns/prints expected column sets.
- Documentation:
  - Document the output model in basics README/docs (brief section on “Rich vs plain summaries”), including how LLMs/MCP can trigger plain mode (e.g., env var or parameter).
