import numpy as np
from math import sqrt
from pyrsm import ifelse


def varprop(x, na=True):
    """
    Calculate the variance for a proportion

    Usage:
    varprop(x)

    Arguments:
    x	Numeric variable with only values 0 and 1
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
    x	Numeric variable with only values 0 and 1
    na  Drop missing values before calculating (True or False)

    Examples:
    seprop([0, 1, 1, 1, 0, 0, 0])
    """
    x = np.array(x)
    if na:
        x = x[np.isnan(x) == False]
    return sqrt(varprop(x, na=False) / len(x))
