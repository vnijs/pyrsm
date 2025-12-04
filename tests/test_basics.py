import numpy as np
import pandas as pd
import polars as pl

from pyrsm.basics.correlation import correlation
from pyrsm.basics.cross_tabs import cross_tabs

df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, 1, 2, 2, 10]})
dfp = pl.DataFrame(df.copy())


def test_correlation_pandas():
    c = correlation(df)
    assert c.cr[1, 0].round(3) == -0.493, "Correlations incorrect"
    df_nan = df.copy()
    df_nan.loc[4, "x"] = np.nan
    c = correlation(df_nan)
    assert c.cr[1, 0].round(3) == -0.567, "Correlations with np.nan incorrect"


def test_correlation_polars():
    c = correlation(dfp)
    assert c.cr[1, 0].round(3) == -0.493, "Correlations incorrect"
    ## looks like an issue converting null to nan in pandas
    # df_nan = dfp
    # df_nan[4, "x"] = None
    # c = correlation(df_nan)
    # assert c.cr[1, 0].round(3) == -0.567, "Correlations with np.nan incorrect"


def test_crosstab():
    ct = cross_tabs(df, "x", "y")
    # Use polars syntax - filter first row (x=0) and check values
    first_row = ct.expected.filter(pl.col("x") == "0")
    expected_vals = [1.714286, 1.714286, 0.571429, 4.0]
    # Check each column except "x"
    numeric_cols = [c for c in ct.expected.columns if c != "x"]
    actual_vals = [first_row[c].item() for c in numeric_cols]
    for actual, expected in zip(actual_vals, expected_vals):
        assert round(actual, 6) == expected, "Cross tab expected values incorrect"


if __name__ == "__main__":
    test_correlation_pandas()
    test_correlation_polars()
    test_crosstab()
