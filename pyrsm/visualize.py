import math
import matplotlib.pyplot as plt
import pandas as pd
from .utils import ifelse
from typing import Tuple, List


def distr_plot(df, nint=25, cols: List = None, **kwargs):
    """
    Plot histograms for numeric variables and frequency plots for categorical.
    variables. Columns of type integer with less than 25 unique values will be
    treated as categorical. To change this behavior, increase or decrease the
    value of the 'nint' argument

    Parameters
    ----------
    df : Pandas dataframe
    nint: int
        The number of unique values in a series of type integer below which the
        series will be treated as a categorical variable
    cols: A list of column names which indicate the subset of variables whose distribution needs to be plotted
    **kwargs : Named arguments to be passed to the pandas plotting methods
    """
    if cols != None:
        df = df[cols]

    fig, axes = plt.subplots(
        math.ceil(df.shape[1] / 2), 2, figsize=(10, 1.5 * df.shape[1])
    )
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    row = 0
    for i, c in enumerate(df.columns):
        s = df[c]
        j = ifelse(i % 2 == 0, 0, 1)
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique() < nint:
            s.value_counts(sort=False).plot.bar(
                ax=axes[row, j], title=c, rot=0, color="slateblue", **kwargs
            )
        elif pd.api.types.is_numeric_dtype(s.dtype):
            s.plot.hist(ax=axes[row, j], title=c, rot=0, color="slateblue", **kwargs)
        elif pd.api.types.is_categorical_dtype(s.dtype):
            s.value_counts(sort=False).plot.bar(
                ax=axes[row, j], title=c, rot=0, color="slateblue", **kwargs
            )
        else:
            print(f"No plot for {c} (type {s.dtype})")

        if j == 1:
            row += 1

    if df.shape[1] % 2 != 0:
        fig.delaxes(axes[row][1])  # remove last empty plot

    plt.show()


def scatter(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    _, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.scatter(df[col1], df[col2])

    plt.show()
