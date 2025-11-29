# Plotting Migration Guide: matplotlib/seaborn → plotnine

## Overview

This guide documents the migration of pyrsm plotting from matplotlib/seaborn to plotnine, with optional plotly support for interactive visualizations.

## Migration Status

### ✅ Completed
- `single_mean.py` - Histogram with CI reference lines

### 🚧 In Progress
- Additional classes in pyrsm/basics/

### ⏳ Planned
- `single_prop.py`
- `compare_props.py`
- `compare_means.py`
- `cross_tabs.py`
- `goodness.py`
- `correlation.py` (dual approach: simple heatmap + complex matrix)

## Architecture

### Core Components

1. **`pyrsm/basics/plotting_utils.py`**: Shared utilities
   - `PlotTheme`: Theme management (modern, publication, minimal, classic)
   - `ReferenceLine`: Helper for CI bounds, comparison values, significance levels
   - `PlotExporter`: Consistent plot export functionality

2. **Dual Backend Support**: Each class supports:
   - `backend="plotnine"`: Static, publication-ready (default)
   - `backend="plotly"`: Interactive, exploratory

3. **Theme System**: Consistent styling across all plots

## Migration Pattern

### Step 1: Update Imports

**Before:**
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
```

**After:**
```python
from plotnine import aes, geom_*, ggplot, labs
import pyrsm.basics.plotting_utils as pu
```

### Step 2: Refactor plot() Method

**New signature:**
```python
def plot(
    self,
    plots: Literal["plot_type1", "plot_type2"] = "default",
    theme: Literal["modern", "publication", "minimal", "classic"] = "modern",
    backend: Literal["plotnine", "plotly"] = "plotnine",
):
```

**Structure:**
```python
def plot(self, plots="hist", theme="modern", backend="plotnine"):
    if backend == "plotly":
        return self._plot_plotly(plots)

    # plotnine implementation
    plot_data = prepare_data()

    p = (
        ggplot(plot_data, aes(...))
        + geom_xxx(...)
        + labs(...)
        + pu.PlotTheme.get_theme(theme)
    )

    # Add reference lines using utilities
    ref_lines = pu.ReferenceLine.ci_vlines(...)
    for line in ref_lines:
        p = p + line

    return p

def _plot_plotly(self, plot_type):
    import plotly.graph_objects as go
    # plotly implementation
    return fig
```

## Common Migrations

### Histograms

**matplotlib:**
```python
self.data[var].plot.hist(bins=30, color="slateblue")
plt.vlines(x=values, ymin=0, ymax=max, colors=["r", "k"], linestyles=["solid", "dashed"])
```

**plotnine:**
```python
(
    ggplot(data, aes(x=var))
    + geom_histogram(bins=30, fill="slateblue", alpha=0.7, color="white")
    + pu.ReferenceLine.ci_vlines(mean, ci_lower, ci_upper, comp_value)
)
```

### Bar Charts

**seaborn:**
```python
sns.barplot(data=data, x="category", y="value", hue="group")
```

**plotnine:**
```python
(
    ggplot(data, aes(x="category", y="value", fill="group"))
    + geom_col(position="dodge")
)
```

### Scatter with Regression

**seaborn:**
```python
sns.regplot(data=data, x="x", y="y", scatter_kws={"alpha": 0.3})
```

**plotnine:**
```python
(
    ggplot(data, aes(x="x", y="y"))
    + geom_point(alpha=0.3)
    + geom_smooth(method="lm", color="blue")
)
```

### Box Plots

**seaborn:**
```python
sns.boxplot(data=data, x="group", y="value")
```

**plotnine:**
```python
(
    ggplot(data, aes(x="group", y="value"))
    + geom_boxplot()
)
```

### Density Plots

**seaborn:**
```python
sns.kdeplot(data=data, x="value", hue="group")
```

**plotnine:**
```python
(
    ggplot(data, aes(x="value", color="group"))
    + geom_density()
)
```

## Testing Approach

### Visual Comparison Tests

Location: `tests/test_plot_migration.py`

```python
def test_xxx_migration():
    # Create object
    obj = ClassName(data, ...)

    def old_plot(ax):
        # matplotlib/seaborn code
        ...

    def new_plot():
        return obj.plot(backend="plotnine")

    PlotComparison.save_comparison(old_plot, new_plot, "test_name")
```

Run: `python tests/test_plot_migration.py`

Output: Side-by-side comparison images in `tests/plot_comparisons/`

## Benefits

### Plotnine
- Consistent grammar of graphics (ggplot2)
- Clean, modern aesthetics by default
- Theme system for customization
- Better composability
- Easier to maintain

### Plotly (Optional)
- Interactive exploration
- Tooltips with exact values
- Zoom/pan capabilities
- Seamless Quarto integration
- Great for dashboards

## Special Cases

### Correlation Matrix (correlation.py)

Due to complexity, dual approach:
1. **`plot_heatmap()`**: Simple correlation heatmap (plotnine)
2. **`plot_matrix()`**: Complex multi-panel matrix (keep matplotlib)

This balances modern aesthetics with practical complexity management.

## Migration Checklist

For each class:
- [ ] Update imports
- [ ] Refactor `plot()` method signature
- [ ] Implement plotnine version
- [ ] Add optional plotly version
- [ ] Create visual comparison test
- [ ] Generate and review comparison images
- [ ] Update docstrings
- [ ] Create example notebook

## Questions?

See `examples/basics/basics-single-mean-plotting.qmd` for a complete working example with both backends.
