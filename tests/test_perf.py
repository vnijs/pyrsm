import pandas as pd
import numpy as np
from pyrsm.perf import (
    calc_qnt,
    gains_tab,
    lift_tab,
    lift_plot,
    gains_plot,
    profit_plot,
    ROME_plot,
    evalbin,
)

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
    assert all(
        ret["cum_resp"] == np.array([6, 14, 20, 27, 29, 30, 30, 30, 30, 30])
    ), "Incorrect calculation of cum_resp with x1"


def test_calc_qnt_rev():
    ret = calc_qnt(df, "response", "yes", "x2", qnt=10)
    assert all(
        ret["cum_resp"] == np.array([6, 14, 20, 27, 29, 30, 30, 30, 30, 30])
    ), "Incorrect calculation of cum_resp with x2"


def test_gains_tab():
    ret = gains_tab(df, "response", "yes", "x1", qnt=10)
    assert all(
        ret["cum_gains"].values.round(3)
        == np.array([0.0, 0.2, 0.467, 0.667, 0.9, 0.967, 1.0, 1.0, 1.0, 1.0, 1.0])
    ), "Incorrect calculation of cum_gains"


def test_lift_tab():
    ret = lift_tab(df, "response", "yes", "x1", qnt=10)
    assert all(
        ret["cum_lift"].values.round(3)
        == np.array([2.0, 2.333, 2.222, 2.25, 1.933, 1.667, 1.429, 1.25, 1.111, 1.0])
    ), "Incorrect calculation of cum_lift"


def test_evalbin():
    dct = {"train": df[df.training == 1], "test": df[df.training == 0]}
    ret = evalbin(dct, "response", "yes", "x1", cost=1, margin=2, dec=3)

    assert ret.shape == (2, 18), "Incorrect dimensions"
    assert all(
        ret.loc[0, "profit":"AUC"] == [5, 1.0, 0.116, 0.538, 0.897]
    ), "Errors for profit:AUC in row 0"
    assert all(
        ret.loc[1, "profit":"AUC"] == [-2, 1.0, -0.143, 0.7, 0.845]
    ), "Errors for profit:AUC in row 0"


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
