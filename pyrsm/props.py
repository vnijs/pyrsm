import polars as pl
from plotnine import aes, geom_col, geom_hline, ggplot, labs, theme_bw

from pyrsm.utils import check_dataframe


def prop_calc(df: pl.DataFrame, group: str, rvar: str, lev: str) -> pl.DataFrame:
    """
    Calculate proportions by a grouping variable

    Parameters
    ----------
    df : Polars dataframe
    group : str
        Name of variable in the dataframe to group by
    rvar : str
        Response variable to calculate proportions for
    lev : str
        Name of the 'success' level in rvar

    Returns
    -------
    A Polars dataframe with the grouping variable and the proportion
    of level "lev" in "rvar" for each value of the grouping variable
    """
    df = check_dataframe(df)
    return (
        df.select([group, rvar])
        .with_columns((pl.col(rvar) == lev).cast(pl.Int32).alias("rvar_int"))
        .group_by(group)
        .agg(pl.col("rvar_int").mean().alias(rvar))
    )


def prop_plot(
    df: pl.DataFrame,
    group: str,
    rvar: str,
    lev: str,
    breakeven: float | None = None,
    color: str = "slateblue",
):
    """
    Plot proportions by a grouping variable

    Parameters
    ----------
    df : Polars dataframe
    group : str
        Name of variable in the dataframe to group by
    rvar : str
        Response variable to calculate proportions for
    lev : str
        Name of the 'success' level in rvar
    breakeven : float or None
        If numeric a horizontal line will be added at the specified breakeven point
    color : str
        Color to use for bars

    Returns
    -------
    ggplot object
    """
    dfp = prop_calc(df, group, rvar, lev)

    p = (
        ggplot(dfp, aes(x=group, y=rvar))
        + geom_col(fill=color)
        + labs(y=f"Proportion of {rvar} equal to '{lev}'")
        + theme_bw()
    )

    if breakeven is not None:
        p = p + geom_hline(yintercept=breakeven, linetype="dashed")

    return p
