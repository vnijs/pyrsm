import numpy as np
from pyrsm.bins import xtile, bincode


def test_xtile():
    x = np.array(range(10))
    bins = xtile(x, 5)
    assert all(
        bins == np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    ), "Incorrect bins returned"


def test_xtile_rev():
    x = np.array(range(10))
    bins = xtile(x, 5, rev=True)
    assert all(
        bins == np.array([5, 5, 4, 4, 3, 3, 2, 2, 1, 1])
    ), "Incorrect reversed bins returned"


def test_xtile_nan():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, np.NaN])
    bins = xtile(x, 5)
    assert all(
        bins.iloc[:9] == np.array([1, 1, 2, 2, 3, 4, 4, 5, 5])
    ), "Incorrect bins with NaN returned"
    assert np.isnan(bins.iloc[-1]), "No missing value returned"


def test_bincode_nan():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, np.NaN])
    breaks = np.quantile(x[np.isnan(x) == False], np.array(range(0, 6)) / 5)
    bins = bincode(x, breaks)
    assert all(
        bins[:9] == np.array([1, 1, 2, 2, 3, 4, 4, 5, 5])
    ), "Incorrect bins with NaN returned"
    assert np.isnan(bins[-1]), "No missing value returned"
