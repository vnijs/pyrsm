from pyrsm.basics import correlation
import numpy as np
import pandas as pd

df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, 1, 2, 2, 10]})


def test_correlation():
    c = correlation(df)
    assert c.cr[1, 0].round(3) == -0.493, "Correlations incorrect"
    df_nan = df.copy()
    df_nan.loc[4, "x"] = np.NaN
    c = correlation(df_nan)
    assert c.cr[1, 0].round(3) == -0.567, "Correlations with np.NaN incorrect"
