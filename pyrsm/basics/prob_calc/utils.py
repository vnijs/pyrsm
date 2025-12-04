from math import ceil, floor

import numpy as np
import polars as pl
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_area,
    geom_col,
    geom_vline,
    labs,
    theme_bw,
    theme,
    scale_fill_manual,
    scale_x_continuous,
)


def iround(x, dec):
    return x if not isinstance(x, float) else round(x, dec)


def pretty_print_summary(summary_dict):
    """Print a summary dictionary in a formatted way."""
    for k, v in summary_dict.items():
        if v is not None:
            print(f"{k}: {v}")


def check(lb, ub, plb, pub):
    if (lb is not None and ub is not None and lb > ub) or (
        plb is not None and pub is not None and plb > pub
    ):
        raise ValueError("Please ensure the lower bound is smaller than the upper bound")


def plot_discrete(x_range, y_range, lb, ub, title=""):
    """
    Create a discrete distribution bar plot using plotnine.

    Parameters
    ----------
    x_range : array-like
        X values (discrete values)
    y_range : array-like
        Y values (probabilities)
    lb : float or None
        Lower bound for highlighting
    ub : float or None
        Upper bound for highlighting
    title : str
        Plot title

    Returns
    -------
    plotnine.ggplot
        The plot object
    """
    # Create DataFrame
    df = pl.DataFrame({"x": x_range, "prob": y_range})

    # Determine fill colors based on bounds
    if lb is not None and ub is not None:
        # Between bounds: slateblue, outside: salmon
        df = df.with_columns(
            pl.when((pl.col("x") >= lb) & (pl.col("x") <= ub))
            .then(pl.lit("in_range"))
            .otherwise(pl.lit("out_range"))
            .alias("fill_type")
        )
    elif lb is not None:
        # >= lb: slateblue, < lb: salmon
        df = df.with_columns(
            pl.when(pl.col("x") >= lb)
            .then(pl.lit("in_range"))
            .otherwise(pl.lit("out_range"))
            .alias("fill_type")
        )
    elif ub is not None:
        # <= ub: slateblue, > ub: salmon
        df = df.with_columns(
            pl.when(pl.col("x") <= ub)
            .then(pl.lit("in_range"))
            .otherwise(pl.lit("out_range"))
            .alias("fill_type")
        )
    else:
        # All slateblue
        df = df.with_columns(pl.lit("in_range").alias("fill_type"))

    p = (
        ggplot(df, aes(x="x", y="prob", fill="fill_type"))
        + geom_col(alpha=0.7, width=0.8)
        + scale_fill_manual(values={"in_range": "slateblue", "out_range": "salmon"})
        + labs(title=title, x="", y="Probability")
        + theme_bw()
        + theme(legend_position="none")
    )

    # Add x-axis breaks if few values
    if len(x_range) <= 20:
        p = p + scale_x_continuous(breaks=list(x_range))

    return p


def plot_continuous(x_range, y_range, lb, ub, title=""):
    """
    Create a continuous distribution plot using plotnine.

    Parameters
    ----------
    x_range : array-like
        X values
    y_range : array-like
        Y values (density)
    lb : float or None
        Lower bound for shading
    ub : float or None
        Upper bound for shading
    title : str
        Plot title

    Returns
    -------
    plotnine.ggplot
        The plot object
    """
    # Convert to numpy arrays if needed
    x_range = np.array(x_range)
    y_range = np.array(y_range)

    # Create DataFrame with shading regions
    df = pl.DataFrame({"x": x_range, "y": y_range})

    # Determine which region each point belongs to
    if lb is not None and ub is not None:
        df = df.with_columns(
            pl.when((pl.col("x") >= lb) & (pl.col("x") <= ub))
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_in"),
            pl.when((pl.col("x") < lb) | (pl.col("x") > ub))
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_out"),
        )
    elif lb is not None:
        df = df.with_columns(
            pl.when(pl.col("x") >= lb)
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_in"),
            pl.when(pl.col("x") < lb)
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_out"),
        )
    elif ub is not None:
        df = df.with_columns(
            pl.when(pl.col("x") <= ub)
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_in"),
            pl.when(pl.col("x") > ub)
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_out"),
        )
    else:
        df = df.with_columns(
            pl.col("y").alias("y_in"),
            pl.lit(0.0).alias("y_out"),
        )

    # Build the plot
    p = (
        ggplot(df, aes(x="x"))
        + geom_area(aes(y="y_in"), fill="slateblue", alpha=0.6)
        + geom_area(aes(y="y_out"), fill="salmon", alpha=0.6)
        + geom_line(aes(y="y"), color="black", size=0.5)
        + labs(title=title, x="", y="Density")
        + theme_bw()
    )

    # Add vertical lines at bounds
    if lb is not None:
        p = p + geom_vline(xintercept=lb, linetype="dashed", color="black", size=0.5)
    if ub is not None:
        p = p + geom_vline(xintercept=ub, linetype="dashed", color="black", size=0.5)

    return p
