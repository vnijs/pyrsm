from pyrsm import xtile, seprop
import numpy as np


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
        bins == np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, -999])
    ), "Incorrect bins with NaN returned"


def test_seprop():
    assert (
        seprop([1, 1, 1, 0, 0, 0]) == 0.2041241452319315
    ), "Proportion standard error incorrect"
