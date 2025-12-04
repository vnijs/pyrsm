# GPT51 PDP Plot Plan (sklearn + statsmodels)

## Goals
- Keep `pred_plot_sk` as-is and add a new `pdp_sk` that offers PDP-like coverage (distribution-averaged) while retaining categorical handling (one plot with all levels, no dummy explosion).
- Add `pdp_sm` as the statsmodels analogue (building on `pred_plot_sm`) with the same coverage options.
- Provide side-by-side visuals and timing info: plot panels showing `pred_plot_sk` vs `pdp_sk` (and `pred_plot_sm` vs `pdp_sm`) plus runtime annotations.
- Improve interaction visuals: avoid sklearn’s default PDP grids; use line-based interactions (10-level slices) and optional heatmaps.

## Function Designs
### pdp_sk
- Modes: `mode="fast"` (current `pred_plot_sk` style) and `mode="pdp"` (distribution-averaged).
- Numeric grids: quantile-based limits (`minq/maxq`, default 0.05/0.95) with `grid_resolution` (default 50–100). For speed, cap unique count.
- Categorical grids: include all observed levels; keep one panel per original categorical variable.
- PDP mode: for each grid value, sample rows (`n_sample` or `frac`, default min(2000, n)) and replace target feature(s), then predict and average. Batch predictions. For tree estimators, optionally use sklearn’s `partial_dependence` fast path and recombine dummy levels to a single plot.
- Interactions:
  - num–num: default to 10 evenly spaced values of one var → 10 lines for the other var; optional heatmap (geom_tile) toggle.
  - num–cat: lines per category across numeric grid.
  - cat–cat: line/point plot with all combos; avoid one plot per dummy.
- Outputs: plotnine plot object and underlying grid data (for tests/LLMs). Annotate runtime in a subtitle/caption.

### pdp_sm
- Similar interface to `pdp_sk`, but using statsmodels predictions. Support classification/regression if applicable.
- Use the same grid construction, sampling, and plotting patterns as `pdp_sk` to keep parity.

## Testing & Benchmarks
- Unit tests:
  - Shape/content: `pdp_sk(mode="pdp")` produces one categorical plot with all levels; interaction outputs match expected grid sizes.
  - Small synthetic datasets: compare `pdp_sk(mode="pdp")` averaged predictions to `sklearn.partial_dependence` arrays for numeric features (tolerance-based checks).
  - Ensure `mode="fast"` matches current `pred_plot_sk` outputs (numeric grids, categorical coverage).
- Visual regression:
  - Save PNGs side-by-side: `pred_plot_sk` vs `pdp_sk`; `pred_plot_sm` vs `pdp_sm`; include runtime text in the plot.
  - Interaction plots: verify line-based interaction (10-level slices) and optional heatmap render.
- Performance:
  - Measure runtime for common estimators (tree, linear/logistic, xgboost) on small/medium data; display runtime in plot captions.
  - Confirm sampling keeps PDP mode competitive with sklearn PDP.

## Integration Steps
1) Implement `pdp_sk` in `pyrsm/model/visualize.py`:
   - Reuse `sim_prediction` for fast mode; add distribution-averaged grid/sampling path.
   - Add runtime capture and subtitle annotation.
   - Keep categorical aggregation and interaction plotting options (lines default, heatmap optional).
2) Implement `pdp_sm` alongside `pred_plot_sm`:
   - Mirror `pdp_sk` API; handle statsmodels predict, weights if needed.
3) Add tests and fixtures:
   - Synthetic numeric + categorical datasets; tree and linear models.
   - PNG outputs for visual comparison saved under `tests/plot_comparisons/pdp/`.
   - Timing assertions (non-zero, reasonable upper bound).
4) Docs:
   - Update docstrings and a short guide section noting `mode`, sampling controls, interaction display options, and categorical handling.
5) Optional: expose a flag to force sklearn’s `partial_dependence` for tree models, but post-process dummy levels into a single categorical panel to avoid fragmented plots.
