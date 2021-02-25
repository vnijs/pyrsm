from pyrsm.stats import (
    varprop,
    seprop,
    weighted_mean,
    weighted_sd,
    scale_df,
)
import numpy as np
import pandas as pd


def test_varprop():
    assert varprop([1, 1, 1, 0, 0, 0]) == 0.25, "Proportion standard error incorrect"


def test_seprop():
    assert (
        seprop([1, 1, 1, 0, 0, 0]) == 0.2041241452319315
    ), "Proportion standard error incorrect"


# create example df and wt vector for testing
df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, 1, 2, 2, 10]})
wt = np.array([1, 10, 1, 10, 1, 10, 1])


def test_weighted_mean():
    assert all(
        weighted_mean(df, wt).round(5) == np.array([0.61765, 1.617650])
    ), "Weighted means incorrect"


def test_weighted_sd():
    assert all(
        weighted_sd(df, wt).round(5) == np.array([0.48596, 1.53421])
    ), "Weighted standard deviations incorrect"


def test_scale_df():
    assert all(
        scale_df(df, ddof=1).round(5).loc[0, ["x", "y"]].values
        == np.array([-0.40089, -0.10984])
    ), "Scaled pandas dataframe incorrect"
    assert all(
        scale_df(df, ddof=1).round(5).loc[1, ["x", "y"]].values
        == np.array([0.53452, -0.26362])
    ), "Scaled pandas dataframe incorrect"


def test_weighted_scale_df():
    assert all(
        scale_df(df, wt, ddof=0).round(5).loc[0, ["x", "y"]].values
        == np.array([-0.63549, 0.12461])
    ), "Weighted scaled pandas dataframe incorrect"
    assert all(
        scale_df(df, wt, ddof=0).round(5).loc[1, ["x", "y"]].values
        == np.array([0.3934, -0.20129])
    ), "Weighted scaled pandas dataframe incorrect"


# def test_correlation():
#     cr, cp = correlation(df, prn=False)
#     assert cr[1, 0].round(3) == -0.493, "Correlations incorrect"
#     df_nan = df.copy()
#     df_nan.loc[4, "x"] = np.NaN
#     cr, cp = correlation(df_nan, prn=False)
#     assert cr[1, 0].round(3) == -0.567, "Correlations with np.NaN incorrect"
