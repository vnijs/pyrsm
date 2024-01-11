from pyrsm.basics.compare_means import compare_means
from pyrsm.basics.correlation import correlation
from pyrsm.basics.cross_tabs import cross_tabs
import numpy as np
import pandas as pd
import polars as pl

df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, 1, 2, 2, 10]})
dfp = pl.DataFrame(df.copy())
dfp
df


# def test_compare_means():
#     cm = compare_means({"df": df}, var1="x", var2="y")
#     assert cm.
#     # assert c.cr[1, 0].round(3) == -0.493, "Correlations incorrect"
# df_nan = df.copy()
# df_nan.loc[4, "x"] = np.NaN
# c = correlation(df_nan)
# assert c.cr[1, 0].round(3) == -0.567, "Correlations with np.NaN incorrect"


def test_correlation_pandas():
    c = correlation(df)
    assert c.cr[1, 0].round(3) == -0.493, "Correlations incorrect"
    df_nan = df.copy()
    df_nan.loc[4, "x"] = np.NaN
    c = correlation(df_nan)
    assert c.cr[1, 0].round(3) == -0.567, "Correlations with np.NaN incorrect"


def test_correlation_polars():
    c = correlation(dfp)
    assert c.cr[1, 0].round(3) == -0.493, "Correlations incorrect"
    ## looks like an issue converting null to NaN in pandas
    # df_nan = dfp
    # df_nan[4, "x"] = None
    # c = correlation(df_nan)
    # assert c.cr[1, 0].round(3) == -0.567, "Correlations with np.NaN incorrect"


def test_crosstab():
    ct = cross_tabs(df, "x", "y")
    assert all(
        ct.expected.iloc[0, :].round(6) == [1.714286, 1.714286, 0.571429, 4.0000]
    ), "Cross tab expected values incorrect"


if __name__ == "__main__":
    test_correlation_pandas()
    test_correlation_polars()
    test_crosstab()
