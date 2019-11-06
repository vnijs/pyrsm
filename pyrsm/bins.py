import numpy as np
import pandas as pd
import sys


def bincode(x, breaks, rev=False, right=True, include_lowest=True):
    """Port of R's .bincode function"""
    n = len(x)
    if any(np.isnan(x)):
        code = np.array([0.0] * n)
    else:
        code = np.array([0] * n)
    nb = len(breaks)
    nb1 = nb - 1
    lft = right is False

    # check if breaks are sorted
    try:
        if sum(breaks[1:] > breaks[-1]) > 0:
            raise ValueError
    except ValueError:
        print("Error: Breaks are not sorted")
        sys.exit(1)

    for i in range(n):
        if np.isnan(x[i]):
            code[i] = np.NaN
            print(f"Entry {i} is a missing value")
        else:
            lo = 0
            hi = nb1
            if (
                x[i] < breaks[lo]
                or breaks[hi] < x[i]
                or (x[i] == breaks[np.where(lft, hi, lo)] and include_lowest is False)
            ):
                next
            else:
                while hi - lo >= 2:
                    new = int(round((hi + lo) / 2, 0))
                    if x[i] > breaks[new] or (lft and x[i] == breaks[new]):
                        lo = new
                    else:
                        hi = new
                code[i] = lo + 1

    return code


def xtile(x, n=5, rev=False):
    """
    Split a numeric series into a number of bins and return a series of bin numbers

    Usage:
    xtile(x, n = 5, rev = FALSE, type = 7)

    Arguments:
    x	Numeric variable
    n	number of bins to create
    rev	Reverse the order of the bin numbers

    Examples:
    xtile(np.array(range(10)), 5)
    xtile(np.array(range(10)), 5, rev=True)
    """
    x = np.array(x)
    breaks = np.quantile(x[np.isnan(x) == False], np.array(range(0, n + 1)) / n)
    if len(np.unique(breaks)) == len(breaks):
        bins = np.array(pd.cut(x, breaks, include_lowest=True, labels=False)) + 1
    else:
        bins = bincode(x, breaks, rev)

    if rev is True:
        bins = (n + 1) - bins

    return bins
