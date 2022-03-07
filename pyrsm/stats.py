import numpy as np
import pandas as pd
from math import sqrt
from pyrsm.utils import ifelse, setdiff


def varprop(x, na=True):
    """
    Calculate the variance for a proportion

    Parameters
    ----------
    x : List, numpy array, or pandas series
        Numeric variable with only values 0 and 1
    na : bool
        Drop missing values before calculating (True or False)

    Returns
    -------
    float
        Calculated variance for a proportion based on a vector of 0 and 1 values

    Examples
    --------
    varprop([0, 1, 1, 1, 0, 0, 0])
    """

    p = ifelse(na, np.nanmean(x), np.mean(x))
    return p * (1 - p)


def sdrop(x, na=True):
    """
    Calculate the standard deviation for a proportion

    Parameters
    ----------
    x : List, numpy array, or pandas series
        Numeric variable with only values 0 and 1
    na : bool
        Drop missing values before calculating (True or False)

    Returns
    -------
    float
        Calculated standard deviation for a proportion based on a vector of 0 and 1 values

    Examples
    --------
    sdprop([0, 1, 1, 1, 0, 0, 0])
    """

    return sqrt(varprop(x, na=na))


def seprop(x, na=True):
    """
    Calculate the standard error for a proportion

    Parameters
    ----------
    x : List, numpy array, or pandas series
        Numeric variable with only values 0 and 1
    na : bool
        Drop missing values before calculating (True or False)

    Returns
    -------
    float
        Calculated variance for a proportion based on a vector of 0 and 1 values

    Examples
    --------
    seprop([0, 1, 1, 1, 0, 0, 0])
    """
    x = np.array(x)
    if na:
        x = x[np.isnan(x) == False]
    return sqrt(varprop(x, na=na) / len(x))


def weighted_sd(df, wt, ddof=0):
    """
    Calculate the weighted standard deviation for a Pandas dataframe

    Parameters
    ----------
    df : Pandas dataframe
        All columns in the dataframe are expected to be numeric
    wt : List, pandas series, or numpy array
        Weights to use during calculation. The length of the vector should be the same as the number of rows in the df
    ddof : int, default 0
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. The default value 0, is
        the same as used by Numpy and sklearn for StandardScaler

    Returns
    -------
    Numpy array
        Array of weighted standard deviations for each column in df

    Examples
    --------
    df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0]})
    wt = [1, 10, 1, 10, 1, 10, 1]
    weighted_sd(df, wt)
    """

    def wsd(x, wt):
        wtm = wt / wt.sum()
        wm = np.average(x, axis=0, weights=wtm)
        wts = wt / (wt.sum() - ddof)
        return sqrt((wts * (x - wm) ** 2).sum())

    wt = np.array(wt)
    return df.apply(lambda col: wsd(col, wt), axis=0).values


def weighted_mean(df, wt):
    """
    Calculate the weighted mean for a Pandas dataframe

    Parameters
    ----------
    df : Pandas dataframe
        All columns in the dataframe are expected to be numeric
    wt : List, pandas series, or numpy array
        Weights to use during calculation. The length of the vector should be the same as the number of rows in the df

    Returns
    -------
    Numpy array
        Array of weighted means for each column in df

    Examples
    --------
    df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0]})
    wt = [1, 10, 1, 10, 1, 10, 1]
    weighted_mean(df, wt)
    """
    return np.average(df.values, weights=np.array(wt), axis=0)


def scale_df(df, wt=None, sf=2, excl=None, train=None, ddof=0):
    """
    Scale the numeric variables in a Pandas dataframe

    Parameters
    ----------
    df : Pandas dataframe with numeric variables
    wt : Pandas series or None
        Weights to use during scaling. The length of the vector should
        be the same as the number of rows in the df
    sf : float
        Scale factor to use (default is 2)
    excl : None or list
        Provide list of column names to exclude when applying standardization
    train : bool
        A series of True and False values. Values that are True are
        used to calculate the mean and standard deviation
    ddof : int, default 0
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. The default value 0, is
        the same as used by Numpy and sklearn for StandardScaler

    Returns
    -------
    Pandas dataframe with all numeric variables standardized

    Examples
    --------
    df = pd.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0]})
    df = scale_df(df)
    """
    df = df.copy()
    isNum = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col].dtype)]
    if excl is not None:
        isNum = setdiff(isNum, excl)
    dfs = df[isNum]
    if train is None:
        train = np.array([True] * df.shape[0])
    if wt is None:
        df[isNum] = (dfs - dfs[train].mean().values) / (
            sf * dfs[train].std(ddof=ddof).values
        )
    else:
        wt = np.array(wt)
        df[isNum] = (dfs - weighted_mean(dfs[train], wt[train])) / (
            sf * weighted_sd(dfs[train], wt[train], ddof=ddof)
        )
    return df
