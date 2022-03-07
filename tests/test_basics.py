from pyrsm.basics import correlation, cross_tabs
import numpy as np
import pandas as pd

df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, 1, 2, 2, 10]})

# ct.expected.iloc[0, :].round(6) == [1.714286, 1.714286, 0.571429, 4.0000]


def test_correlation():
    c = correlation(df)
    assert c.cr[1, 0].round(3) == -0.493, "Correlations incorrect"
    df_nan = df.copy()
    df_nan.loc[4, "x"] = np.NaN
    c = correlation(df_nan)
    assert c.cr[1, 0].round(3) == -0.567, "Correlations with np.NaN incorrect"


def test_crosstab():
    ct = cross_tabs(df, "x", "y")
    assert all(
        ct.expected.iloc[0, :].round(6) == [1.714286, 1.714286, 0.571429, 4.0000]
    ), "Cross tab expected values incorrect"
