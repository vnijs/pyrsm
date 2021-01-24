import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyrsm import xtile
from pyrsm.utils import ifelse


def calc_qnt(df, rvar, lev, pred, qnt=10):
    """
    Create quantiles and calculate input to use for lift and gains charts

    Parameters
    ----------
    df : Pandas dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create

    Returns
    -------
    Pandas dataframe
        Response metrics per quantile. Used as input for lift and gains charts
    """

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


def calc_dec(df, rvar, lev, pred, qnt=10):
    """Deprecated function. Use calc_qnt instead"""
    print("The 'calc_dec' function is deprecated. Use 'calc_qnt' instead")
    return calc_dec(df, rvar, lev, pred, qnt=10)


def calc(df, rvar, lev, pred, qnt=10):
    """Deprecated function. Use calc_qnt instead"""
    print("The 'calc' function is deprecated. Use 'calc_qnt' instead")
    return None


def gains_tab(df, rvar, lev, pred, qnt=10):
    """
    Calculate cumulative gains using the cum_resp column

    Parameters
    ----------
    df : Pandas dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create

    Returns
    -------
    Pandas dataframe
        Gains measures per quantile. Input for gains chart
    """

    df = calc_qnt(df, rvar, lev, pred, qnt=qnt)
    df["cum_gains"] = df["cum_resp"] / df["cum_resp"].iloc[-1]
    df0 = pd.DataFrame({"cum_prop": [0], "cum_gains": [0]})
    df = pd.concat([df0, df], sort=False)
    df.index = range(df.shape[0])
    return df[["cum_prop", "cum_gains"]]


def lift_tab(df, rvar, lev, pred, qnt=10):
    """
    Calculate cumulative lift using the cum_resp and the cum_obs column

    Parameters
    ----------
    df : Pandas dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create

    Returns
    -------
    Pandas dataframe
        Lift measures per quantile. Input for lift chart
    """

    df = calc_qnt(df, rvar, lev, pred, qnt=qnt)
    df["cum_resp_rate"] = df["cum_resp"] / df["cum_obs"]
    df["cum_lift"] = df["cum_resp_rate"] / df["cum_resp_rate"].iloc[-1]
    df.index = range(df.shape[0])
    return df[["cum_prop", "cum_lift"]]


def confusion(df, rvar, lev, pred, cost=1, margin=2):
    """
    Calculate TP, FP, TN, FN, and contact

    Parameters
    ----------
    df : Pandas dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    TP : int
        Number of True Positive predictions
    FP : int
        Number of False Positive predictions
    TN : int
        Number of True Negative predictions
    FN : int
        Number of False Negative predictions
    contact: float
        Proportion of cases to act on based on the cost/margin ratio
    """

    break_even = cost / margin
    gtbe = df[pred] > break_even
    pos = df[rvar] == lev
    TP = np.where(gtbe & pos, 1, 0).sum()
    FP = np.where((gtbe == True) & (pos == False), 1, 0).sum()
    TN = np.where((gtbe == False) & (pos == False), 1, 0).sum()
    FN = np.where((gtbe == False) & (pos == True), 1, 0).sum()
    contact = (TP + FP) / (TP + FP + TN + FN)
    return TP, FP, TN, FN, contact


def profit_max(df, rvar, lev, pred, cost=1, margin=2):
    """
    Calculate the maximum profit using a dataframe as input

    Parameters
    ----------
    df : Pandas dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    float
        Measure of optimal performace (e.g., profit) based on the specified cost and margin information
    """

    TP, FP, TN, FN, contact = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    return margin * TP - cost * (TP + FP)


def profit(pred, rvar, lev, cost=1, margin=2):
    """
    Calculate the maximum profit using series as input. Provides the same results as profit_max

    Parameters
    ----------
    pred : Pandas series
        Column from a Pandas dataframe with model predictions
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model prediction
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    float
        Measure of optimal performace (e.g., profit) based on the specified cost and margin information
    """

    break_even = cost / margin
    TP = ((rvar == lev) & (pred > break_even)).sum()
    FP = ((rvar != lev) & (pred > break_even)).sum()
    return margin * TP - cost * (TP + FP)


def ROME_max(df, rvar, lev, pred, cost=1, margin=2):
    """
    Calculate the maximum Return on Marketing Expenditures

    Parameters
    ----------
    df : Pandas dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    float
        Measure of optimal performace based on the specified cost and margin information
    """

    TP, FP, TN, FN, contact = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    profit = margin * TP - cost * (TP + FP)
    return profit / (cost * (TP + FP))


def ROME(pred, rvar, lev, cost=1, margin=2):
    """
    Calculate the maximum Return on Marketing Expenditures using series as input. Provides the same results as ROME_max

    Parameters
    ----------
    pred : Pandas series
        Column from a Pandas dataframe with model predictions
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model prediction
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    float
        Measure of optimal performace (e.g., profit) based on the specified cost and margin information
    """

    break_even = cost / margin
    TP = ((rvar == lev) & (pred > break_even)).sum()
    FP = ((rvar != lev) & (pred > break_even)).sum()
    profit = margin * TP - cost * (TP + FP)
    return profit / (cost * (TP + FP))


def profit_tab(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """
    Calculate table with profit per quantile

    Parameters
    ----------
    df : Pandas dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    Pandas dataframe
        Profit per quantile. Input for profit chart
    """

    df = calc_qnt(df, rvar, lev, pred, qnt=qnt)
    df["cum_profit"] = margin * df["cum_resp"] - cost * df["cum_obs"]
    df0 = pd.DataFrame({"cum_prop": [0], "cum_profit": [0]})
    df = pd.concat([df0, df], sort=False)
    df.index = range(df.shape[0])
    return df[["cum_prop", "cum_profit"]]


def ROME_tab(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """
    Calculate table with Return on Marketing Expenditures per quantile

    Parameters
    ----------
    df : Pandas dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    Pandas dataframe
        ROME quantile. Input for ROME chart
    """

    df = calc_qnt(df, rvar, lev, pred, qnt=qnt)
    df["cum_profit"] = margin * df["cum_resp"] - cost * df["cum_obs"]
    cum_cost = cost * df["cum_obs"]
    df["ROME"] = (margin * df["cum_resp"] - cum_cost) / cum_cost
    df.index = range(df.shape[0])
    return df[["cum_prop", "ROME"]]


def profit_plot(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """
    Plot a profit curve

    Parameters
    ----------
    df : Pandas dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    Seaborn object
        Plot of profits per quantile

    Examples
    --------
    profit_plot(df, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    profit_plot(df, "buyer", "yes", ["pred_a", "pred_b"], cost=0.5, margin=6)
    dct = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    profit_plot(dct, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    """

    dct = ifelse(type(df) is dict, df, {"": df})
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1 or len(dct.keys()) > 1, "predictor", None)
    cnf = [
        confusion(dct[k], rvar, lev, p, cost=cost, margin=margin)[-1]
        for k in dct.keys()
        for p in pred
    ]
    df = [
        profit_tab(dct[k], rvar, lev, p, qnt=qnt, cost=cost, margin=margin).assign(
            predictor=p + ifelse(k == "", k, f" ({k})")
        )
        for k in dct.keys()
        for p in pred
    ]
    df = pd.concat(df)
    fig = sns.lineplot(x="cum_prop", y="cum_profit", data=df, hue=group, marker="o")
    fig.set(ylabel="Profit", xlabel="Proportion of customers")
    fig.axhline(1, linestyle="--", linewidth=1)
    [fig.axvline(l, linestyle="--", linewidth=1) for l in cnf]
    return fig


def ROME_plot(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """
    Plot a ROME curve

    Parameters
    ----------
    df : Pandas dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    Seaborn object
        Plot of ROME per quantile

    Examples
    --------
    ROME_plot(df, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    ROME_plot(df, "buyer", "yes", ["pred_a", "pred_b"], cost=0.5, margin=6)
    dct = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    ROME_plot(dct, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    """
    dct = ifelse(type(df) is dict, df, {"": df})
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1 or len(dct.keys()) > 1, "predictor", None)
    rd = [
        ROME_tab(dct[k], rvar, lev, p, qnt=qnt, cost=cost, margin=margin).assign(
            predictor=p + ifelse(k == "", k, f" ({k})")
        )
        for k in dct.keys()
        for p in pred
    ]
    rd = pd.concat(rd)
    fig = sns.lineplot(x="cum_prop", y="ROME", data=rd, hue=group, marker="o")
    fig.set(
        ylabel="Return on Marketing Expenditures (ROME)",
        xlabel="Proportion of customers",
    )
    fig.axhline(0, linestyle="--", linewidth=1)
    return fig


def gains_plot(df, rvar, lev, pred, qnt=10):
    """
    Plot a cumulative gains curve

    Parameters
    ----------
    df : Pandas dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    Seaborn object
        Plot of gaines per quantile

    Examples
    --------
    gains_plot(df, "buyer", "yes", "pred_a")
    gains_plot(df, "buyer", "yes", ["pred_a", "pred_b"], qnt=20)
    dct = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    gains_plot(dct, "buyer", "yes", "pred_a")
    """
    dct = ifelse(type(df) is dict, df, {"": df})
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1 or len(dct.keys()) > 1, "predictor", None)
    rd = [
        gains_tab(dct[k], rvar, lev, p, qnt=qnt).assign(
            predictor=p + ifelse(k == "", k, f" ({k})")
        )
        for k in dct.keys()
        for p in pred
    ]
    rd = pd.concat(rd)
    fig = sns.lineplot(x="cum_prop", y="cum_gains", data=rd, hue=group, marker="o")
    fig.set(ylabel="Cumulative gains", xlabel="Proportion of customers")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    return fig


def lift_plot(df, rvar, lev, pred, qnt=10):
    """
    Plot a cumulative lift chart

    Parameters
    ----------
    df : Pandas dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    Seaborn object
        Plot of lift per quantile

    Examples
    --------
    lift_plot(df, "buyer", "yes", "pred_a")
    lift_plot(df, "buyer", "yes", ["pred_a", "pred_b"], qnt=20)
    lift = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    lift_plot(dct, "buyer", "yes", "pred_a")
    """
    dct = ifelse(type(df) is dict, df, {"": df})
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1 or len(dct.keys()) > 1, "predictor", None)
    rd = [
        lift_tab(dct[k], rvar, lev, p, qnt=qnt).assign(
            predictor=p + ifelse(k == "", k, f" ({k})")
        )
        for k in dct.keys()
        for p in pred
    ]
    rd = pd.concat(rd)
    fig = sns.lineplot(x="cum_prop", y="cum_lift", data=rd, hue=group, marker="o")
    fig.axhline(1, linestyle="--", linewidth=1)
    fig.set(ylabel="Cumulative lift", xlabel="Proportion of customers")
    return fig
