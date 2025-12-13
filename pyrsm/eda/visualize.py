"""
visualize() - Create plots using plotnine.

Supports: dist (histogram/bar), density, scatter, line, bar, box, violin
Options: color, fill, facet, smooth (lm/loess), jitter, bins, alpha, size

Examples:
    import pyrsm as rsm

    rsm.eda.visualize(df, x="price")                          # histogram for numeric
    rsm.eda.visualize(df, x="cut")                            # bar for categorical
    rsm.eda.visualize(df, x="price", geom="density")          # density plot
    rsm.eda.visualize(df, x="carat", y="price")               # scatter plot
    rsm.eda.visualize(df, x="carat", y="price", smooth="lm")  # with regression line
    rsm.eda.visualize(df, x="cut", y="price", geom="box")     # box plot
    rsm.eda.visualize(df, x="price", facet="cut")             # faceted histograms
"""

from typing import Optional, Union
import polars as pl

# Geom configurations: required aesthetics and defaults
GEOM_CONFIG = {
    "dist": {
        "required": ["x"],
        "defaults": {"bins": 30, "fill": "slateblue", "alpha": 0.7},
    },
    "hist": {  # Alias for dist (numeric)
        "required": ["x"],
        "defaults": {"bins": 30, "fill": "slateblue", "alpha": 0.7},
    },
    "density": {
        "required": ["x"],
        "defaults": {"fill": "slateblue", "alpha": 0.5},
    },
    "scatter": {
        "required": ["x", "y"],
        "defaults": {"alpha": 0.7, "size": 2, "nobs": 1000},
    },
    "bar": {
        "required": ["x"],
        "defaults": {"fill": "slateblue", "alpha": 0.8},
    },
    "line": {
        "required": ["x", "y"],
        "defaults": {"size": 1},
    },
    "box": {
        "required": ["x", "y"],
        "defaults": {"fill": "slateblue", "alpha": 0.7},
    },
    "violin": {
        "required": ["x", "y"],
        "defaults": {"fill": "slateblue", "alpha": 0.7},
    },
}


def _is_categorical(df: pl.DataFrame, col: str) -> bool:
    """Check if column is categorical (string/enum or low-cardinality int)."""
    dtype = df.schema.get(col)
    if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum):
        return True
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        # Treat integers with few unique values as categorical
        n_unique = df.select(pl.col(col).n_unique()).item()
        return n_unique <= 20
    return False


def visualize(
    df: Union[pl.DataFrame, pl.LazyFrame],
    x: str,
    y: Optional[str] = None,
    geom: Optional[str] = None,
    color: Optional[str] = None,
    fill: Optional[str] = None,
    shape: Optional[str] = None,
    group: Optional[str] = None,
    linetype: Optional[str] = None,
    bins: Optional[int] = None,
    alpha: Optional[float] = None,
    size: Optional[Union[int, float]] = None,
    position: Optional[str] = None,
    smooth: Optional[str] = None,
    jitter: bool = False,
    facet: Optional[str] = None,
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    title: Optional[str] = None,
    nobs: int = 1000,
    agg: Optional[str] = None,
):
    """
    Create a plot using plotnine.

    Args:
        df: Polars DataFrame or LazyFrame
        x: Column name for x-axis
        y: Column name for y-axis (required for scatter, line, box, violin)
        geom: Plot type: dist, hist, density, scatter, bar, line, box, violin
              Default: scatter if y provided, dist otherwise
        color: Column name for color aesthetic or literal color
        fill: Column name for fill aesthetic or literal color
        shape: Column name for shape aesthetic
        group: Column name for grouping
        linetype: Column name for linetype aesthetic
        bins: Number of bins for histogram (default: 30)
        alpha: Transparency (0-1)
        size: Point/line size
        position: Bar position: "stack" or "dodge"
        smooth: Add smooth line to scatter: "lm", "loess", or "true"
        jitter: Add jitter to scatter plot points
        facet: Column for facet_wrap
        facet_row: Row faceting variable (for facet_grid)
        facet_col: Column faceting variable (for facet_grid)
        title: Plot title
        nobs: Max observations for scatter plots (default: 1000, -1 for all)
        agg: Aggregation function for bar/scatter plots with categorical x:
             "mean", "median", "sum", "min", "max". For bar plots, aggregates y
             by x. For scatter plots with categorical x, adds a line showing
             the aggregated value per category.

    Returns:
        plotnine ggplot object

    Examples:
        >>> rsm.eda.visualize(df, x="price")  # histogram
        >>> rsm.eda.visualize(df, x="carat", y="price")  # scatter
        >>> rsm.eda.visualize(df, x="carat", y="price", color="cut")
        >>> rsm.eda.visualize(df, x="carat", y="price", smooth="lm")
        >>> rsm.eda.visualize(df, x="cut", y="price", geom="bar", agg="mean")  # mean bar
        >>> rsm.eda.visualize(df, x="cut", y="price", geom="scatter", agg="mean")  # scatter with mean line
    """
    from plotnine import (
        ggplot,
        aes,
        geom_histogram,
        geom_bar,
        geom_col,
        geom_density,
        geom_point,
        geom_jitter,
        geom_line,
        geom_boxplot,
        geom_violin,
        geom_smooth,
        stat_summary,
        facet_wrap,
        facet_grid,
        labs,
        theme_bw,
    )

    # Aggregation function mapping using lambdas with Polars
    AGG_FUNCS = {
        "mean": lambda x: x.mean(),
        "median": lambda x: pl.Series(x).median(),
        "sum": lambda x: pl.Series(x).sum(),
        "min": lambda x: pl.Series(x).min(),
        "max": lambda x: pl.Series(x).max(),
    }

    # Validate agg argument
    if agg is not None and agg not in AGG_FUNCS:
        available = ", ".join(sorted(AGG_FUNCS.keys()))
        raise ValueError(f"Unknown agg: {agg}. Available: {available}")

    # Convert LazyFrame to DataFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Determine geom type
    if geom is None:
        geom = "scatter" if y else "dist"

    if geom not in GEOM_CONFIG:
        available = ", ".join(sorted(GEOM_CONFIG.keys()))
        raise ValueError(f"Unknown geom: {geom}. Available: {available}")

    # Validate required aesthetics
    config = GEOM_CONFIG[geom]
    if "x" in config["required"] and not x:
        raise ValueError(f"x is required for geom={geom}")
    if "y" in config["required"] and not y:
        raise ValueError(f"y is required for geom={geom}")

    # Build aesthetics dict (column mappings only)
    aes_kwargs = {"x": x}
    if y:
        aes_kwargs["y"] = y

    # Add color/fill if they're column names
    if color and color in df.columns:
        aes_kwargs["color"] = color
    if fill and fill in df.columns:
        aes_kwargs["fill"] = fill
    if shape and shape in df.columns:
        aes_kwargs["shape"] = shape
    if group and group in df.columns:
        aes_kwargs["group"] = group
    if linetype and linetype in df.columns:
        aes_kwargs["linetype"] = linetype

    # Build geom kwargs (non-aesthetic params)
    geom_kwargs = {}

    # Apply defaults then overrides
    for key, default in config["defaults"].items():
        geom_kwargs[key] = default

    if bins is not None:
        geom_kwargs["bins"] = bins
    if alpha is not None:
        geom_kwargs["alpha"] = alpha
    if size is not None:
        geom_kwargs["size"] = size

    # Handle literal colors (not column names)
    if color and color not in df.columns:
        geom_kwargs["color"] = color
    if fill and fill not in df.columns:
        geom_kwargs["fill"] = fill

    # Sample data for scatter plots if needed
    nobs_caption = None
    if geom == "scatter" and nobs != -1 and len(df) > nobs:
        df = df.sample(n=nobs, seed=1234)
        nobs_caption = f"nobs={nobs} used"

    # Build base plot
    p = ggplot(df, aes(**aes_kwargs))

    # Add geom layer
    if geom in ("dist", "hist"):
        if _is_categorical(df, x):
            # Categorical: use bar chart
            bar_kwargs = {k: v for k, v in geom_kwargs.items() if k != "bins"}
            p = p + geom_bar(**bar_kwargs)
        else:
            # Numeric: use histogram
            hist_bins = geom_kwargs.pop("bins", 30)
            p = p + geom_histogram(bins=hist_bins, **geom_kwargs)

    elif geom == "density":
        p = p + geom_density(**geom_kwargs)

    elif geom == "scatter":
        scatter_kwargs = {k: v for k, v in geom_kwargs.items() if k != "nobs"}
        if jitter:
            p = p + geom_jitter(width=0.2, height=0, **scatter_kwargs)
        else:
            p = p + geom_point(**scatter_kwargs)

        # Add smooth line if requested
        if smooth:
            if smooth == "lm":
                p = p + geom_smooth(method="lm", se=True, alpha=0.2)
            elif smooth == "loess":
                p = p + geom_smooth(method="loess", se=True, alpha=0.2)
            elif smooth in ("true", "True", True):
                p = p + geom_smooth(se=True, alpha=0.2)

        # Add aggregation line for categorical x
        if agg and _is_categorical(df, x):
            agg_func = AGG_FUNCS[agg]
            p = p + stat_summary(fun_y=agg_func, fun_ymin=agg_func, fun_ymax=agg_func, geom="crossbar", color="blue", size=1)

    elif geom == "bar":
        pos = position or "stack"
        bar_kwargs = {k: v for k, v in geom_kwargs.items() if k != "position"}
        if agg and y:
            # Aggregated bar plot: use stat_summary
            agg_func = AGG_FUNCS[agg]
            p = p + geom_bar(stat="summary", fun_y=agg_func, position=pos, **bar_kwargs)
        else:
            p = p + geom_bar(stat="count", position=pos, **bar_kwargs)

    elif geom == "line":
        # For line plots, group by color if specified
        if "color" in aes_kwargs and "group" not in aes_kwargs:
            aes_kwargs["group"] = aes_kwargs["color"]
            p = ggplot(df, aes(**aes_kwargs))
        line_size = geom_kwargs.pop("size", 1)
        line_kwargs = {k: v for k, v in geom_kwargs.items() if k != "fill"}
        p = p + geom_line(size=line_size, **line_kwargs)

    elif geom == "box":
        p = p + geom_boxplot(**geom_kwargs)

    elif geom == "violin":
        p = p + geom_violin(**geom_kwargs)

    # Add faceting
    if facet:
        p = p + facet_wrap(f"~{facet}")
    elif facet_row or facet_col:
        row = facet_row or "."
        col = facet_col or "."
        p = p + facet_grid(f"{row}~{col}")

    # Add labels and theme
    x_lab = x
    y_lab = y or ""
    if geom in ("dist", "hist", "bar") and not y:
        y_lab = "Count"
    if geom == "density" and not y:
        y_lab = "Density"
    if agg and y and geom == "bar":
        y_lab = f"{agg.capitalize()} of {y}"

    p = p + labs(x=x_lab, y=y_lab, title=title or "", caption=nobs_caption) + theme_bw()

    return p
