import numpy as np
import pandas as pd
import polars as pl

from pyrsm.basics.single_mean import single_mean


def test_single_mean_pandas_basic():
    df = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
    sm = single_mean(df, var="values", comp_value=3)

    assert sm.n == 5
    assert sm.n_missing == 0
    assert sm.mean == 3
    assert sm.diff == 0
    assert sm.df == 4
    assert sm.t_val == 0
    assert sm.p_val == 1
    assert sm.me > 0


def test_single_mean_polars_with_missing():
    df = pl.DataFrame({"values": [10.0, 11.0, None, 9.0]})
    sm = single_mean(df, var="values", comp_value=10)

    assert sm.n == 4
    assert sm.n_missing == 1
    assert sm.mean == 10.0
    assert sm.diff == 0
    assert np.isfinite(sm.sd)
    assert np.isfinite(sm.se)
    assert sm.df == 2


def test_single_mean_alt_hyp_greater():
    df = pd.DataFrame({"values": [5, 6, 7, 8, 9]})
    sm = single_mean(df, var="values", comp_value=4, alt_hyp="greater", conf=0.9)

    assert sm.mean == 7
    assert sm.diff == 3
    assert sm.p_val < 0.01
    assert sm.t_val > 0
