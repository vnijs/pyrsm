import pandas as pd
import polars as pl
import numpy as np
import pytest
from pyrsm.model.perf import (
    calc_qnt,
    gains_tab,
    lift_tab,
    lift_plot,
    gains_plot,
    profit_plot,
    ROME_plot,
    evalbin,
    auc,
)


@pytest.fixture(scope="module")
def perf_test_data():
    """Create test data in both pandas and polars formats."""
    np.random.seed(1234)
    nr = 100
    pdf = pd.DataFrame()
    pdf["x1"] = np.random.uniform(0, 1, nr)
    pdf["x2"] = 1 - pdf.x1
    pdf["response_prob"] = np.random.uniform(0, 1, nr)
    pdf["response"] = np.where((pdf["x1"] > 0.5) & (pdf["response_prob"] > 0.5), "yes", "no")
    pdf["training"] = np.concatenate([np.ones(80), np.zeros(20)])
    pdf["rnd_response"] = np.random.choice(["yes", "no"], nr)
    plf = pl.from_pandas(pdf)
    return pdf, plf


# Legacy global df for backward compatibility with existing tests
np.random.seed(1234)
nr = 100
df = pd.DataFrame()
df["x1"] = np.random.uniform(0, 1, nr)
df["x2"] = 1 - df.x1
df["response_prob"] = np.random.uniform(0, 1, nr)
df["response"] = np.where((df["x1"] > 0.5) & (df["response_prob"] > 0.5), "yes", "no")
df["training"] = np.concatenate([np.ones(80), np.zeros(20)])
df["rnd_response"] = np.random.choice(["yes", "no"], nr)


def test_calc_qnt():
    ret = calc_qnt(df, "response", "yes", "x1", qnt=10)
    assert ret["cum_resp"].to_list() == [6, 14, 20, 27, 29, 30, 30, 30, 30, 30], \
        "Incorrect calculation of cum_resp with x1"


def test_calc_qnt_rev():
    ret = calc_qnt(df, "response", "yes", "x2", qnt=10)
    assert ret["cum_resp"].to_list() == [6, 14, 20, 27, 29, 30, 30, 30, 30, 30], \
        "Incorrect calculation of cum_resp with x2"


def test_calc_qnt_polars(perf_test_data):
    """Test calc_qnt with polars input."""
    _, plf = perf_test_data
    ret = calc_qnt(plf, "response", "yes", "x1", qnt=10)
    assert ret["cum_resp"].to_list() == [6, 14, 20, 27, 29, 30, 30, 30, 30, 30], \
        "Incorrect calculation of cum_resp with polars input"


def test_gains_tab():
    ret = gains_tab(df, "response", "yes", "x1", qnt=10)
    expected = [0.0, 0.2, 0.467, 0.667, 0.9, 0.967, 1.0, 1.0, 1.0, 1.0, 1.0]
    actual = [round(v, 3) for v in ret["cum_gains"].to_list()]
    assert actual == expected, "Incorrect calculation of cum_gains"


def test_lift_tab():
    ret = lift_tab(df, "response", "yes", "x1", qnt=10)
    expected = [2.0, 2.333, 2.222, 2.25, 1.933, 1.667, 1.429, 1.25, 1.111, 1.0]
    actual = [round(v, 3) for v in ret["cum_lift"].to_list()]
    assert actual == expected, "Incorrect calculation of cum_lift"


def test_evalbin():
    dct = {"train": df[df.training == 1], "test": df[df.training == 0]}
    ret = evalbin(dct, "response", "yes", "x1", cost=1, margin=2, dec=3)

    assert ret.shape == (2, 18), "Incorrect dimensions"
    row0 = ret.row(0, named=True)
    assert [row0["profit"], row0["index"], row0["ROME"], row0["contact"], row0["AUC"]] == [
        5, 1.0, 0.116, 0.538, 0.897
    ], "Errors for profit:AUC in row 0"
    row1 = ret.row(1, named=True)
    assert [row1["profit"], row1["index"], row1["ROME"], row1["contact"], row1["AUC"]] == [
        -2, 1.0, -0.143, 0.7, 0.845
    ], "Errors for profit:AUC in row 1"


def test_lift_plot_single():
    fig = lift_plot(df, "response", "yes", "x1", qnt=10)


def test_lift_plot_mult():
    fig = lift_plot(df, "response", "yes", ["x1", "x2"], qnt=10)


def test_gains_plot_single():
    fig = gains_plot(df, "response", "yes", "x1", qnt=10)


def test_gains_plot_mult():
    fig = gains_plot(df, "response", "yes", ["x1", "x2"], qnt=10)


def test_profit_plot_single():
    fig = profit_plot(df, "response", "yes", "x1", qnt=10)


def test_profit_plot_mult():
    fig = profit_plot(df, "response", "yes", ["x1", "x2"], qnt=10, contact=True)


def test_rome_plot_single():
    fig = ROME_plot(df, "response", "yes", "x1", qnt=10)


def test_ROME_plot_mult():
    fig = ROME_plot(df, "response", "yes", ["x1", "x2"], qnt=10)


# ---- Tests with polars input ----


def test_gains_tab_polars(perf_test_data):
    """Test gains_tab with polars input."""
    _, plf = perf_test_data
    ret = gains_tab(plf, "response", "yes", "x1", qnt=10)
    expected = [0.0, 0.2, 0.467, 0.667, 0.9, 0.967, 1.0, 1.0, 1.0, 1.0, 1.0]
    actual = [round(v, 3) for v in ret["cum_gains"].to_list()]
    assert actual == expected, "Incorrect calculation of cum_gains with polars"


def test_lift_tab_polars(perf_test_data):
    """Test lift_tab with polars input."""
    _, plf = perf_test_data
    ret = lift_tab(plf, "response", "yes", "x1", qnt=10)
    expected = [2.0, 2.333, 2.222, 2.25, 1.933, 1.667, 1.429, 1.25, 1.111, 1.0]
    actual = [round(v, 3) for v in ret["cum_lift"].to_list()]
    assert actual == expected, "Incorrect calculation of cum_lift with polars"


def test_evalbin_polars(perf_test_data):
    """Test evalbin with polars input."""
    _, plf = perf_test_data
    dct = {
        "train": plf.filter(pl.col("training") == 1),
        "test": plf.filter(pl.col("training") == 0),
    }
    ret = evalbin(dct, "response", "yes", "x1", cost=1, margin=2, dec=3)

    assert ret.shape == (2, 18), "Incorrect dimensions with polars"
    row0 = ret.row(0, named=True)
    assert [row0["profit"], row0["index"], row0["ROME"], row0["contact"], row0["AUC"]] == [
        5, 1.0, 0.116, 0.538, 0.897
    ], "Errors for profit:AUC in row 0 with polars"


def test_lift_plot_polars(perf_test_data):
    """Test lift_plot with polars input."""
    _, plf = perf_test_data
    fig = lift_plot(plf, "response", "yes", "x1", qnt=10)


def test_gains_plot_polars(perf_test_data):
    """Test gains_plot with polars input."""
    _, plf = perf_test_data
    fig = gains_plot(plf, "response", "yes", "x1", qnt=10)


def test_profit_plot_polars(perf_test_data):
    """Test profit_plot with polars input."""
    _, plf = perf_test_data
    fig = profit_plot(plf, "response", "yes", "x1", qnt=10)


def test_ROME_plot_polars(perf_test_data):
    """Test ROME_plot with polars input."""
    _, plf = perf_test_data
    fig = ROME_plot(plf, "response", "yes", "x1", qnt=10)


# ---- AUC tests with different input types ----


@pytest.fixture(scope="module")
def auc_test_data():
    """Create test data for AUC in numpy, pandas, and polars formats."""
    np.random.seed(42)
    n = 100
    pred = np.random.uniform(0, 1, n)
    rvar = np.where(pred + np.random.normal(0, 0.3, n) > 0.5, "yes", "no")
    return pred, rvar


def test_auc_numpy(auc_test_data):
    """Test auc with numpy arrays."""
    pred, rvar = auc_test_data
    result = auc(rvar, pred, lev="yes")
    assert isinstance(result, float)
    assert 0.5 <= result <= 1.0
    # Store for comparison
    test_auc_numpy.result = result


def test_auc_pandas(auc_test_data):
    """Test auc with pandas Series."""
    pred, rvar = auc_test_data
    pred_pd = pd.Series(pred)
    rvar_pd = pd.Series(rvar)
    result = auc(rvar_pd, pred_pd, lev="yes")
    assert isinstance(result, float)
    assert 0.5 <= result <= 1.0
    # Store for comparison
    test_auc_pandas.result = result


def test_auc_polars(auc_test_data):
    """Test auc with polars Series."""
    pred, rvar = auc_test_data
    pred_pl = pl.Series("pred", pred)
    rvar_pl = pl.Series("rvar", rvar)
    result = auc(rvar_pl, pred_pl, lev="yes")
    assert isinstance(result, float)
    assert 0.5 <= result <= 1.0
    # Store for comparison
    test_auc_polars.result = result


def test_auc_all_types_match(auc_test_data):
    """Test that auc produces identical results for numpy, pandas, and polars inputs."""
    pred, rvar = auc_test_data

    # Numpy
    result_np = auc(rvar, pred, lev="yes")

    # Pandas
    result_pd = auc(pd.Series(rvar), pd.Series(pred), lev="yes")

    # Polars
    result_pl = auc(pl.Series("rvar", rvar), pl.Series("pred", pred), lev="yes")

    assert np.isclose(result_np, result_pd), f"numpy ({result_np}) != pandas ({result_pd})"
    assert np.isclose(result_np, result_pl), f"numpy ({result_np}) != polars ({result_pl})"
    assert np.isclose(result_pd, result_pl), f"pandas ({result_pd}) != polars ({result_pl})"


def test_auc_with_ties(auc_test_data):
    """Test auc handles ties correctly across all input types."""
    # Data with intentional ties
    pred = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.5])
    rvar = np.array(["yes", "no", "yes", "no", "yes", "no", "yes", "no"])

    result_np = auc(rvar, pred, lev="yes")
    result_pd = auc(pd.Series(rvar), pd.Series(pred), lev="yes")
    result_pl = auc(pl.Series("rvar", rvar), pl.Series("pred", pred), lev="yes")

    assert np.isclose(result_np, result_pd)
    assert np.isclose(result_np, result_pl)


if __name__ == "__main__":
    test_calc_qnt()
    test_calc_qnt_rev()
    test_gains_tab()
    test_lift_tab()
    test_evalbin()
    test_lift_plot_single()
    test_lift_plot_mult()
    test_gains_plot_single()
    test_gains_plot_mult()
    test_profit_plot_single()
    test_profit_plot_mult()
    test_rome_plot_single()
    test_ROME_plot_mult()
