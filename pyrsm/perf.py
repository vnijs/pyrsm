import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyrsm import xtile


def calc(df, rvar, lev, pred, qnt=10):
    """Create deciles and calculate input to use for lift and gains charts"""
    df["bins"] = xtile(df[pred], qnt)
    df["rvar_int"] = np.where(df[rvar] == lev, 1, 0)
    perf_df = (
        df.groupby("bins")["rvar_int"].agg(nr_obs="count", nr_resp=sum).reset_index()
    )

    # flip if needed
    if perf_df["nr_resp"].iloc[1] < perf_df["nr_resp"].iloc[-1]:
        perf_df = perf_df.sort_values("bins", ascending=False)

    # resp_rate = perf_df.nr_resp / perf_df.nr_obs
    perf_df["cum_obs"] = np.cumsum(perf_df["nr_obs"])
    perf_df["cum_prop"] = perf_df["cum_obs"] / perf_df["cum_obs"].iloc[-1]
    perf_df["cum_resp"] = np.cumsum(perf_df["nr_resp"])

    return perf_df


def gains(df, rvar, lev, pred, qnt=10):
    """Calculate cumulative gains using the cum_resp column"""
    df = calc(df, rvar, lev, pred, qnt=qnt)
    df["cum_gains"] = df["cum_resp"] / df["cum_resp"].iloc[-1]
    return df[["cum_prop", "cum_gains"]]


def lift(df, rvar, lev, pred, qnt=10):
    # need cum_resp and cum_obs
    """Calculate cumulative lift using the cum_resp and the cum_obs column"""
    df = calc(df, rvar, lev, pred, qnt=qnt)
    df["cum_resp_rate"] = df["cum_resp"] / df["cum_obs"]
    df["cum_lift"] = df["cum_resp_rate"] / df["cum_resp_rate"].iloc[-1]
    return df[["cum_prop", "cum_lift"]]


def confusion(df, rvar, lev, pred, cost=1, margin=2):
    """Calculate TP, FP, TN, FN, and contact"""
    df["rvar_int"] = np.where(df[rvar] == lev, 1, 0)
    break_even = cost / margin
    gtbe = df[pred] > break_even
    pos = df["rvar_int"] == 1
    TP = np.where(gtbe & pos, 1, 0).sum()
    FP = np.where(gtbe & pos == False, 1, 0).sum()
    TN = np.where((gtbe == False) & (pos == False), 1, 0).sum()
    FN = np.where(gtbe == False & pos, 1, 0).sum()
    contact = (TP + FP) / (TP + FP + TN + FN)
    return TP, FP, TN, FN, contact


def profit_max(df, rvar, lev, pred, cost=1, margin=2):
    """Calculate the maximum profit"""
    TP, FP, TN, FN, contact = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    return margin * TP - cost * (TP + FP)


def ROME(df, pred, rvar, lev, cost=1, margin=2):
    """Calculate the Return on Marketing Expenditures"""
    TP, FP, TN, FN, contact = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    profit = margin * TP - cost * (TP + FP)
    return profit / (cost * (TP + FP))


def profit(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """Calculate profits"""
    df = calc(df, rvar, lev, pred, qnt=qnt)
    df["cum_profit"] = margin * df["cum_resp"] - cost * df["cum_obs"]
    return df[["cum_prop", "cum_profit"]]


def profit_plot(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """Plot a profit chart"""
    cnf = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    df = profit(df, rvar, lev, pred, qnt=qnt, cost=cost, margin=margin)
    df0 = pd.DataFrame()
    df0["cum_prop"] = np.array([0.0] * (df.shape[0] + 1))
    df0["cum_prop"][1:] = df["cum_prop"].values
    df0["cum_profit"] = df0["cum_prop"]
    df0["cum_profit"][1:] = df["cum_profit"].values
    plt.clf()
    fig = sns.lineplot(x="cum_prop", y="cum_profit", data=df0, marker="o")
    fig.set(ylabel="Profit", xlabel="Proportion of customers")
    fig.axhline(1)
    fig.axvline(cnf[-1], linestyle="--", linewidth=1)
    plt.show()


def gains_plot(df, rvar, lev, pred, qnt=10):
    """Plot a cumulative gains chart"""
    df = gains(df, rvar, lev, pred, qnt=qnt)
    df0 = pd.DataFrame()
    df0["cum_prop"] = np.array([0.0] * (df.shape[0] + 1))
    df0["cum_gains"] = df0["cum_prop"]
    df0["cum_prop"][1:] = df["cum_prop"].values
    df0["cum_gains"][1:] = df["cum_gains"].values
    plt.clf()
    fig = sns.lineplot(x="cum_prop", y="cum_gains", data=df0, marker="o")
    fig.set(ylabel="Cumulative gains", xlabel="Proportion of customers")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.show()


def lift_plot(df, rvar, lev, pred, qnt=10):
    """Plot a cumulative lift chart"""
    df = lift(df, rvar, lev, pred, qnt=qnt)
    plt.clf()
    fig = sns.lineplot(x="cum_prop", y="cum_lift", data=df, marker="o")
    fig.axhline(1)
    fig.set(ylabel="Cumulative lift", xlabel="Proportion of customers")
    plt.show()
