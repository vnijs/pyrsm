import numpy as np
import seaborn as sns


def prop_calc(df, group, rvar, lev):
    """
    Calculate proportions by a grouping variable

    Arguments:
    df          A data frame
    group       A variable in the data frame to group by
    rvar        The response variable to calculate proportions for
    lev         The level considered 'success'

    Return:
    A data frame with the grouping variable and the proportion of level
        "lev" in "rvar" for each value of the grouping variable
    """
    df["rvar_int"] = np.where(df[rvar] == lev, 1, 0)
    df = df.groupby(group)
    return (
        df.agg(prop=("rvar_int", "mean")).reset_index().rename(columns={"prop": rvar})
    )


def prop_plot(df, group, rvar, lev, breakeven=None):
    """
    Plot proportions by a grouping variable

    Arguments:
    df          A data frame
    group       A variable in the data frame to group by
    rvar        The response variable to calculate proportions for
    lev         The level considered 'success'
    breakeven   If numeric a horizontal line will be added at the
                   specified breakeven point
    """
    dfp = prop_calc(df, group, rvar, lev)
    cn = dfp.columns
    fig = sns.barplot(x=cn[0], y=cn[1], color="slateblue", data=dfp)
    fig.set(ylabel=f"Proportion of {cn[1]} = '{lev}'")
    if breakeven is not None:
        fig.axhline(breakeven, linestyle="dashed", linewidth=0.5)
    return fig
