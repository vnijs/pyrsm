import numpy as np
import pandas as pd
from math import sqrt
from pyrsm import ifelse
from itertools import compress


def varprop(x, na=True):
    """
    Calculate the variance for a proportion

    Usage:
    varprop(x)

    Arguments:
    x   Numeric variable with only values 0 and 1
    na  Drop missing values before calculating (True or False)

    Examples:
    varprop([0, 1, 1, 1, 0, 0, 0])
    """

    p = ifelse(na, np.nanmean(x), np.mean(x))
    return p * (1 - p)


def seprop(x, na=True):
    """
    Calculate the standard error for a proportion

    Usage:
    seprop(x)

    Arguments:
    x   Numeric variable with only values 0 and 1
    na  Drop missing values before calculating (True or False)

    Examples:
    seprop([0, 1, 1, 1, 0, 0, 0])
    """
    x = np.array(x)
    if na:
        x = x[np.isnan(x) == False]
    return sqrt(varprop(x, na=False) / len(x))


def weighted_sd(df, wt):
    """
    Calculate the weighted standard deviation for a pandas data frame

    Usage:
    weighted_sd(df, wt)

    Arguments:
    df  A pandas data frame with numeric variables
    wt  Weights

    Examples:
    df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0]})
    wt = [1, 10, 1, 10, 1, 10, 1]
    weighted_sd(df, wt)
    """

    def wsd(x, wt):
        wt = wt / wt.sum()
        wm = np.average(x, axis=0, weights=wt)
        return sqrt((wt * (x - wm) ** 2).sum())

    wt = np.array(wt)
    return df.apply(lambda col: wsd(col, wt), axis=0).values


def weighted_mean(df, wt):
    """
    Calculate the weighted mean for a pandas data frame

    Usage:
    weighted_mean(df, wt)

    Arguments:
    df  A pandas data frame with numeric variables
    wt  Weights

    Examples:
    df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0]})
    wt = [1, 10, 1, 10, 1, 10, 1]
    weighted_mean(df, wt)
    """
    return np.average(df.values, weights=np.array(wt), axis=0)


def scale_df(df, wt=None, sf=2):
    """
    Scale the numeric variables in a pandas data frame

    Usage:
    scale_df(df)

    Arguments:
    df  A pandas data frame with numeric variables
    wt  Weights
    sf  Scale factor to use (2 is the default)

    Examples:
    df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0]})
    wt = [1, 10, 1, 10, 1, 10, 1]
    weighted_mean(df, wt)
    """

    df = df.copy()
    isNum = [pd.api.types.is_numeric_dtype(df[col]) for col in df.columns]
    isNum = list(compress(df.columns, isNum))
    dfs = df[isNum]
    if wt is None:
        df[isNum] = (dfs - dfs.mean().values) / (sf * dfs.std().values)
    else:
        df[isNum] = (dfs - weighted_mean(dfs, wt)) / (sf * weighted_sd(dfs, wt))
    return df
