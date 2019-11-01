import numpy as np
from math import sqrt


def seprop(x):
    """
    Calculate the standard error for a proportion

    Usage:
    seprop(x)

    Arguments:
    x	Numeric variable with only values 0 and 1

    Examples:
    seprop([0, 1, 1, 1, 0, 0, 0])
    """
    p = np.nanmean(x)
    varprop = p * (1 - p)
    return sqrt(varprop / len(x))
