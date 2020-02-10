import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyrsm import xtile
from pyrsm.utils import ifelse


def calc(df, rvar, lev, pred, qnt=10):
    """Create deciles and calculate input to use for lift and gains charts"""
    df = df.loc[:, (rvar, pred)]
    df["bins"] = xtile(df[pred], qnt)
    df["rvar_int"] = np.where(df[rvar] == lev, 1, 0)
    perf_df = (
        df.groupby("bins")["rvar_int"].agg(nr_obs="count", nr_resp=sum).reset_index()
    )

    # flip if needed
    if perf_df["nr_resp"].iloc[1] < perf_df["nr_resp"].iloc[-1]:
        perf_df = perf_df.sort_values("bins", ascending=False)

    perf_df["cum_obs"] = np.cumsum(perf_df["nr_obs"])
    perf_df["cum_prop"] = perf_df["cum_obs"] / perf_df["cum_obs"].iloc[-1]
    perf_df["cum_resp"] = np.cumsum(perf_df["nr_resp"])
    return perf_df


def gains(df, rvar, lev, pred, qnt=10):
    """Calculate cumulative gains using the cum_resp column"""
    df = calc(df, rvar, lev, pred, qnt=qnt)
    df["cum_gains"] = df["cum_resp"] / df["cum_resp"].iloc[-1]
    df0 = pd.DataFrame({"cum_prop": [0], "cum_gains": [0]})
    df = pd.concat([df0, df], sort=False)
    df.index = range(df.shape[0])
    return df[["cum_prop", "cum_gains"]]


def lift(df, rvar, lev, pred, qnt=10):
    # need cum_resp and cum_obs
    """Calculate cumulative lift using the cum_resp and the cum_obs column"""
    df = calc(df, rvar, lev, pred, qnt=qnt)
    df["cum_resp_rate"] = df["cum_resp"] / df["cum_obs"]
    df["cum_lift"] = df["cum_resp_rate"] / df["cum_resp_rate"].iloc[-1]
    df.index = range(df.shape[0])
    return df[["cum_prop", "cum_lift"]]


def confusion(df, rvar, lev, pred, cost=1, margin=2):
    """Calculate TP, FP, TN, FN, and contact"""
    rvar_int = np.where(df[rvar] == lev, 1, 0)
    break_even = cost / margin
    gtbe = df[pred] > break_even
    pos = rvar_int == 1
    TP = np.where(gtbe & pos, 1, 0).sum()
    FP = np.where((gtbe == True) & (pos == False), 1, 0).sum()
    TN = np.where((gtbe == False) & (pos == False), 1, 0).sum()
    FN = np.where((gtbe == False) & (pos == True), 1, 0).sum()
    contact = (TP + FP) / (TP + FP + TN + FN)
    return TP, FP, TN, FN, contact


def profit_max(df, rvar, lev, pred, cost=1, margin=2):
    """Calculate the maximum profit"""
    TP, FP, TN, FN, contact = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    return margin * TP - cost * (TP + FP)


def ROME_max(df, rvar, lev, pred, cost=1, margin=2):
    """Calculate the maximum Return on Marketing Expenditures"""
    TP, FP, TN, FN, contact = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    profit = margin * TP - cost * (TP + FP)
    return profit / (cost * (TP + FP))


def profit(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """Calculate profits"""
    df = calc(df, rvar, lev, pred, qnt=qnt)
    df["cum_profit"] = margin * df["cum_resp"] - cost * df["cum_obs"]
    df0 = pd.DataFrame({"cum_prop": [0], "cum_profit": [0]})
    df = pd.concat([df0, df], sort=False)
    df.index = range(df.shape[0])
    return df[["cum_prop", "cum_profit"]]


def ROME(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """Calculate the Return on Marketing Expenditures"""
    df = calc(df, rvar, lev, pred, qnt=qnt)
    df["cum_profit"] = margin * df["cum_resp"] - cost * df["cum_obs"]
    cum_cost = cost * df["cum_obs"]
    df["ROME"] = (margin * df["cum_resp"] - cum_cost) / cum_cost
    df.index = range(df.shape[0])
    return df[["cum_prop", "ROME"]]


def profit_plot(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """Plot a profit chart"""
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1, "predictor", None)
    cnf = [confusion(df, rvar, lev, p, cost=cost, margin=margin)[-1] for p in pred]
    df = [
        profit(df, rvar, lev, p, qnt=qnt, cost=cost, margin=margin).assign(predictor=p)
        for p in pred
    ]
    df = pd.concat(df)
    fig = sns.lineplot(x="cum_prop", y="cum_profit", data=df, hue=group, marker="o")
    fig.set(ylabel="Profit", xlabel="Proportion of customers")
    fig.axhline(1, linestyle="--", linewidth=1)
    [fig.axvline(l, linestyle="--", linewidth=1) for l in cnf]
    return fig


def ROME_plot(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """Plot a ROME chart"""
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1, "predictor", None)
    df = [
        ROME(df, rvar, lev, p, qnt=qnt, cost=cost, margin=margin).assign(predictor=p)
        for p in pred
    ]
    df = pd.concat(df)
    fig = sns.lineplot(x="cum_prop", y="ROME", data=df, hue=group, marker="o")
    fig.set(
        ylabel="Return on Marketing Expenditures (ROME)",
        xlabel="Proportion of customers",
    )
    fig.axhline(0, linestyle="--", linewidth=1)
    return fig


def gains_plot(df, rvar, lev, pred, qnt=10):
    """Plot a cumulative gains chart"""
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1, "predictor", None)
    df = [gains(df, rvar, lev, p, qnt=qnt).assign(predictor=p) for p in pred]
    df = pd.concat(df)
    fig = sns.lineplot(x="cum_prop", y="cum_gains", data=df, hue=group, marker="o")
    fig.set(ylabel="Cumulative gains", xlabel="Proportion of customers")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    return fig


def lift_plot(df, rvar, lev, pred, qnt=10):
    """Plot a cumulative lift chart"""
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1, "predictor", None)
    df = [lift(df, rvar, lev, p, qnt=qnt).assign(predictor=p) for p in pred]
    df = pd.concat(df)
    fig = sns.lineplot(x="cum_prop", y="cum_lift", data=df, hue=group, marker="o")
    fig.axhline(1, linestyle="--", linewidth=1)
    fig.set(ylabel="Cumulative lift", xlabel="Proportion of customers")
    return fig
