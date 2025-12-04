# GPT51 Polars + Plotnine Migration Plan

## Objectives
- Make polars the default dataframe everywhere; keep pandas only at library boundaries (e.g., scikit-learn), converting via `pl_df.to_pandas()` immediately before the call.
- Replace all matplotlib/seaborn plotting with plotnine (native polars support), standardize plot creation/outputs, and drop pandas-only plotting paths.
- Minimize numpy usage; prefer polars expressions/Series methods unless SciPy/stat functions require numpy arrays or a polars approach is materially slower.

## Current Usage Inventory (needs conversion)
- Pandas imports: `pyrsm/utils.py`, `pyrsm/stats.py`, `pyrsm/example_data.py`, most `pyrsm/model/*` (regress, logistic, rforest, xgboost, mlp, visualize), `pyrsm/radiant/utils.py`, `pyrsm/radiant/model/model_utils.py`, streamlit/shiny apps, tests, and basics tests. Helpers in `utils.py` (add_description/describe) are pandas-specific.
- Plotting with matplotlib/seaborn: basics (`correlation`, `compare_means`, `compare_props`, `cross_tabs`, `single_prop`, `central_limit_theorem`, `goodness`), props, model plots (regress/logistic/rforest/xgboost/mlp), and shiny/genai tools.
- Explicit pandas-for-plot comments: model modules (`regress`, `logistic`, `rforest`, `xgboost`), basics `compare_props`, `central_limit_theorem`.
- numpy-heavy math: correlation matrices, stat helpers, plotting jitter, etc., across basics and model modules.

## Migration Principles
- Data handling: operate on polars `DataFrame/Series` end-to-end. When third-party APIs require pandas (sklearn, statsmodels), call `.to_pandas()` on the minimal subset right before the API call; avoid early conversion for plotting.
- Plotting: new plotnine helpers should accept polars data directly; return plotnine objects and optional saved images. Centralize theme/labels to keep consistency.
- Output format: reuse the notebook-vs-terminal rendering pattern (see `compare_means` and planned `display_utils`) for summaries built from polars DataFrames.
- Performance: prefer polars expressions; allow numpy only for SciPy calls or when benchmarks show polars is slower.

## Workstreams & File Targets
1) **Shared Utilities**
   - `pyrsm/utils.py`: add polars-native description/metadata helpers; deprecate pandas-only `add_description/describe` or wrap with polars path. Ensure `check_dataframe` returns polars everywhere.
   - New `plotnine_utils` to standardize figure creation, saving, and theme; accept polars data and lazily convert subsets for APIs lacking polars support.
   - `display_utils`: extend to all basics outputs for styled/plain rendering.

2) **Basics module conversions**
   - `compare_means`, `compare_props`, `cross_tabs`, `correlation`, `single_prop`, `central_limit_theorem`, `goodness`, probability calculators: eliminate pandas paths (already partially polars). Replace matplotlib/seaborn plots with plotnine equivalents (scatter/box/density/bar, chi-square visuals, CLT hist/density, prop plots).
   - Replace numpy-centric transforms with polars expressions where feasible (groupby aggs, variance/mean, melt/unpivot via polars).
   - Ensure summaries expose polars DataFrames to the shared renderer; update docstrings/examples to show polars + plotnine usage.

3) **Model layer**
   - `pyrsm/model/regress.py`, `logistic.py`, `rforest.py`, `xgboost.py`, `mlp.py`, `visualize.py`: keep modeling data in polars; convert to pandas only when calling sklearn/xgboost; plotting via plotnine instead of seaborn/matplotlib. Remove pandas DataFrame construction except at API boundary.
   - `pyrsm/model/model.py` and `radiant/model/model_utils.py`: audit helper functions that assume pandas (e.g., `.values`, `.iloc`); refactor to polars equivalents or add boundary conversions.

4) **Data loading & props**
   - `pyrsm/example_data.py`: load datasets as polars by default; optional pandas flag triggers `.to_pandas()`.
   - `pyrsm/props.py`: convert seaborn barplot usage to plotnine; keep data as polars.

5) **Apps (shiny/streamlet/genai)**
   - Replace pandas imports with polars; only convert to pandas where UI components require it. Swap matplotlib plots with plotnine and render as images/HTML in apps.

6) **Tests**
   - Update tests to prefer polars fixtures; only use pandas when asserting boundary conversions. Remove pandas imports where not required. Add plotnine snapshot tests (PNG output) replacing matplotlib baselines.

## Plotnine Conversion Targets (examples)
- Scatter/box/density/bar: use `ggplot(polars_df) + geom_point/boxplot/density/ribbon/bar` with facets as needed.
- Correlation plotting: build pairwise data via polars -> plotnine `geom_point` with `geom_smooth` (method="lm") and facet grid; correlation text via `geom_text`.
- Cross-tabs/props: bar/stacked bar plots with `geom_col` and `position="fill"`; add annotation layers.
- Model diagnostics: residuals, ROC/PR, feature importance via plotnine geoms; avoid pandas melt—use polars `melt`/`select`.

## Performance & Fallbacks
- Benchmark key paths (correlation, compare_means, model preprocessing) after polars refactor; keep numpy fallbacks only if polars is slower with equivalent semantics.
- Where SciPy/statsmodels need numpy arrays, convert minimal columns with `.to_numpy()`; keep upstream data polars.

## Rollout Steps
1) Freeze current behavior with tests: add/expand pytest coverage (including PNG baselines for seaborn/matplotlib plots) capturing current pandas + seaborn/matplotlib outputs across basics and model components.
2) Establish shared helpers (`plotnine_utils`, enhanced `display_utils`, polars-first `utils`) and update `compare_means` to consume them as reference.
3) Convert basics plotting to plotnine module by module, removing pandas/seaborn/matplotlib usage; adjust summaries to polars-only and rerun the frozen tests to ensure parity.
4) Refactor model modules to polars-first, pandas only at sklearn/xgboost call sites; replace plotting with plotnine; rerun tests to confirm unchanged behavior.
5) Update data loading/props and app layers to polars + plotnine; revalidate with existing coverage.
6) Refresh tests/fixtures to polars defaults; keep plotnine PNG baselines current.
7) Document migration notes and known boundary conversions in README/AGENTS for contributors.***
