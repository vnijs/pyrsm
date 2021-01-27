import numpy as np
import seaborn as sns


def prop_calc(df, group, rvar, lev):
    """
    Calculate proportions by a grouping variable

    Parameters
    ----------
    df : Pandas dataframe
    group : str
        Name of variable in the dataframe to group by
    rvar : str
        Response variable to calculate proportions for
    lev : str
        Name of the 'success' level in rvar

    Returns
    -------
    A Pandas dataframe with the grouping variable and the proportion of level
        "lev" in "rvar" for each value of the grouping variable
    """

    df = df.loc[:, (group, rvar)]
    df["rvar_int"] = np.where(df[rvar] == lev, 1, 0)
    df = df.groupby(group)
    return (
        df.agg(prop=("rvar_int", "mean")).reset_index().rename(columns={"prop": rvar})
    )


def prop_plot(
    df, group, rvar, lev, breakeven=None, linewidth=1, color="slateblue", **kwargs
):
    """
    Plot proportions by a grouping variable

    Parameters
    ----------
    df : Pandas dataframe
    group : str
        Name of variable in the dataframe to group by
    rvar : str
        Response variable to calculate proportions for
    lev : str
        Name of the 'success' level in rvar
    breakeven : float or None
        If numeric a horizontal line will be added at the specified breakeven point
    linewidth : float
        Width to use for the breakeven line in the plot
    color : str
        Color to use for bars
    **kwargs : Named arguments to be passed to the seaborn barplot function
    """
    dfp = prop_calc(df, group, rvar, lev)
    cn = dfp.columns
    fig = sns.barplot(x=cn[0], y=cn[1], color=color, data=dfp, **kwargs)
    fig.set(ylabel=f"Proportion of {cn[1]} equal to '{lev}'")
    if breakeven is not None:
        fig.axhline(breakeven, linestyle="dashed", linewidth=linewidth)
    return fig
