import math
import matplotlib.pyplot as plt
import pandas as pd
from pyrsm.utils import ifelse


def distr_plot(df, nint=25, **kwargs):
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
    **kwargs : Named arguments to be passed to the pandas plotting methods
    """
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
