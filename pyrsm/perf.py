import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyrsm import xtile
from pyrsm.utils import ifelse
from sklearn import metrics
from scipy.stats import rankdata


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
    perf_df = df.groupby("bins").rvar_int.agg(nr_obs="count", nr_resp=sum).reset_index()

    # flip if needed
    if perf_df.nr_resp.iloc[0] < perf_df.nr_resp.iloc[-1]:
        perf_df = perf_df.sort_values("bins", ascending=False)

    perf_df["cum_obs"] = np.cumsum(perf_df.nr_obs)
    perf_df["cum_prop"] = perf_df.cum_obs / perf_df.cum_obs.iloc[-1]
    perf_df["cum_resp"] = np.cumsum(perf_df.nr_resp)
    return perf_df


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
    df["cum_gains"] = df.cum_resp / df.cum_resp.iloc[-1]
    df0 = pd.DataFrame({"cum_prop": [0], "cum_gains": [0]})
    df = pd.concat([df0, df], sort=False).reset_index(drop=True)
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
    df["cum_resp_rate"] = df.cum_resp / df.cum_obs
    df["cum_lift"] = df.cum_resp_rate / df.cum_resp_rate.iloc[-1]
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

    if isinstance(pred, list) and len(pred) > 1:
        return "This function can only take one predictor variables at time"

    break_even = cost / margin
    gtbe = df[pred] > break_even
    pos = df[rvar] == lev
    TP = np.where(gtbe & pos, 1, 0).sum()
    FP = np.where(gtbe & np.logical_not(pos), 1, 0).sum()
    TN = np.where(np.logical_not(gtbe) & np.logical_not(pos), 1, 0).sum()
    FN = np.where(np.logical_not(gtbe) & pos, 1, 0).sum()
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


def profit(rvar, pred, lev=1, cost=1, margin=2):
    """
    Calculate the maximum profit using series as input. Provides the same results as profit_max

    Parameters
    ----------
    rvar : Pandas series
        Column from a Pandas dataframe with the response variable
    pred : Pandas series
        Column from a Pandas dataframe with model predictions
    lev : str
        Name of the 'success' level in rvar
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
    Calculate the maximum Return on Marketing Expenditures using series as input.
    Provides the same results as ROME_max

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
    df["cum_profit"] = margin * df.cum_resp - cost * df.cum_obs
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
    df["cum_profit"] = margin * df.cum_resp - cost * df.cum_obs
    cum_cost = cost * df.cum_obs
    df["ROME"] = (margin * df.cum_resp - cum_cost) / cum_cost
    df.index = range(df.shape[0])
    return df[["cum_prop", "ROME"]]


def profit_plot(
    df, rvar, lev, pred, qnt=10, cost=1, margin=2, contact=False, marker="o", **kwargs
):
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
    contact : bool
        Plot a vertical line that shows the optimal contact level. Requires
        that `pred` is a series of probabilities. Values equal to 1 (100% contact)
        will not be plotted
    marker : str
        Marker to use for line plot
    **kwargs : Named arguments to be passed to the seaborn lineplot function

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
    df = [
        profit_tab(dct[k], rvar, lev, p, qnt=qnt, cost=cost, margin=margin).assign(
            predictor=p + ifelse(k == "", k, f" ({k})")
        )
        for k in dct.keys()
        for p in pred
    ]
    df = pd.concat(df).reset_index(drop=True)
    fig = sns.lineplot(
        x="cum_prop", y="cum_profit", data=df, hue=group, marker=marker, **kwargs
    )
    fig.set(ylabel="Profit", xlabel="Proportion of customers")
    fig.axhline(1, linestyle="--", linewidth=1)
    if contact:
        cnf = [
            confusion(dct[k], rvar, lev, p, cost=cost, margin=margin)[-1]
            for k in dct.keys()
            for p in pred
        ]
        prof = [
            profit_max(dct[k], rvar, lev, p, cost=cost, margin=margin)
            for k in dct.keys()
            for p in pred
        ]
        [
            [
                fig.axvline(
                    l, linestyle="--", linewidth=1, color=sns.color_palette()[i]
                ),
                fig.axhline(
                    prof[i], linestyle="--", linewidth=1, color=sns.color_palette()[i]
                ),
            ]
            for i, l in enumerate(filter(lambda x: x < 1, cnf))
        ]
    if len(dct) > 1 or len(pred) > 1:
        fig.legend(title=None)

    return fig


def expected_profit_plot(
    df, rvar, lev, pred, cost=1, margin=2, contact=False, **kwargs
):
    """
    Plot an expected profit curve

    Parameters
    ----------
    df : Pandas dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
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
    contact : bool
        Plot a vertical line that shows the optimal contact level. Requires
        that `pred` is a series of probabilities. Values equal to 1 (100% contact)
        will not be plotted
    **kwargs : Named arguments to be passed to the seaborn lineplot function

    Returns
    -------
    Seaborn object
        Plot of profits per quantile

    Examples
    --------
    expected_profit_plot(df, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    expected_profit_plot(df, "buyer", "yes", ["pred_a", "pred_b"], cost=0.5, margin=6)
    dct = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    expected_profit_plot(dct, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    """
    dct = ifelse(type(df) is dict, df, {"": df})
    pred = ifelse(type(pred) is list, pred, [pred])
    group = ifelse(len(pred) > 1 or len(dct.keys()) > 1, "predictor", None)

    def calc_exp_profit(df, pred, cost, margin):
        prediction = df[pred].sort_values(ascending=False)
        profit = prediction * margin - cost
        return pd.DataFrame(
            {
                "cum_prop": np.arange(1, df.shape[0] + 1) / df.shape[0],
                "cum_profit": np.cumsum(profit),
            }
        )

    df = [
        calc_exp_profit(dct[k], p, cost, margin).assign(
            predictor=p + ifelse(k == "", k, f" ({k})")
        )
        for k in dct.keys()
        for p in pred
    ]
    df = pd.concat(df).reset_index(drop=True)
    fig = sns.lineplot(x="cum_prop", y="cum_profit", data=df, hue=group, **kwargs)
    fig.set(ylabel="Profit", xlabel="Proportion of customers")
    fig.axhline(1, linestyle="--", linewidth=1)
    if contact:
        cnf = [
            confusion(dct[k], rvar, lev, p, cost=cost, margin=margin)[-1]
            for k in dct.keys()
            for p in pred
        ]
        prof = [
            profit_max(dct[k], rvar, lev, p, cost=cost, margin=margin)
            for k in dct.keys()
            for p in pred
        ]
        [
            [
                fig.axvline(
                    l, linestyle="--", linewidth=1, color=sns.color_palette()[i]
                ),
                fig.axhline(
                    prof[i], linestyle="--", linewidth=1, color=sns.color_palette()[i]
                ),
            ]
            for i, l in enumerate(filter(lambda x: x < 1, cnf))
        ]
    if len(dct) > 1 or len(pred) > 1:
        fig.legend(title=None)

    return fig


def ROME_plot(df, rvar, lev, pred, qnt=10, cost=1, margin=2, marker="o", **kwargs):
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
    marker : str
        Marker to use for line plot
    **kwargs : Named arguments to be passed to the seaborn lineplot function

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
    rd = pd.concat(rd).reset_index(drop=True)
    fig = sns.lineplot(
        x="cum_prop", y="ROME", data=rd, hue=group, marker=marker, **kwargs
    )
    fig.set(
        ylabel="Return on Marketing Expenditures (ROME)",
        xlabel="Proportion of customers",
    )
    fig.axhline(0, linestyle="--", linewidth=1)
    if len(dct) > 1 or len(pred) > 1:
        fig.legend(title=None)
    return fig


def gains_plot(df, rvar, lev, pred, qnt=10, marker="o", **kwargs):
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
    marker : str
        Marker to use for line plot
    **kwargs : Named arguments to be passed to the seaborn lineplot function

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
    rd = pd.concat(rd).reset_index(drop=True)
    fig = sns.lineplot(
        x="cum_prop", y="cum_gains", data=rd, hue=group, marker=marker, **kwargs
    )
    fig.set(ylabel="Cumulative gains", xlabel="Proportion of customers")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=plt.cm.Blues(0.7))
    if len(dct) > 1 or len(pred) > 1:
        fig.legend(title=None)
    return fig


def lift_plot(df, rvar, lev, pred, qnt=10, marker="o", **kwargs):
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
        Name, or list, of the column(s) in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    marker : str
        Marker to use for line plot
    **kwargs : Named arguments to be passed to the seaborn lineplot function

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
    rd = pd.concat(rd).reset_index(drop=True)
    fig = sns.lineplot(
        x="cum_prop", y="cum_lift", data=rd, hue=group, marker=marker, **kwargs
    )
    fig.set(ylabel="Cumulative lift", xlabel="Proportion of customers")
    fig.axhline(1, linestyle="--", linewidth=1)
    if len(dct) > 1 or len(pred) > 1:
        fig.legend(title=None)
    return fig


def evalbin(df, rvar, lev, pred, cost=1, margin=2, dec=3):
    """
    Evaluate binary classification models. Calculates TP, FP, TN, FN, contact, total,
    TPR, TNR, precision, Fscore, accuracy, profit, ROME, AUC, kappa, and profit index

    Parameters
    ----------
    df : Pandas dataframe or a dictionary of dataframes with keys to show results for
        multiple model predictions and datasets (training and test)
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column, of list of column names, in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    dec : int
        Number of decimal places to use in rounding

    Examples
    --------
    """

    dct = ifelse(type(df) is dict, df, {"All": df})
    pred = ifelse(type(pred) is list, pred, [pred])

    def calculate_metrics(key, dfm, pm):
        TP, FP, TN, FN, contact = confusion(dfm, rvar, lev, pm, cost, margin)
        total = TN + FN + FP + TP
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        precision = TP / (TP + FP)
        profit = margin * TP - cost * (TP + FP)

        fpr, tpr, thresholds = metrics.roc_curve(dfm[rvar], dfm[pm], pos_label=lev)
        break_even = cost / margin
        gtbe = dfm[pm] > break_even
        pos = dfm[rvar] == lev

        return pd.DataFrame().assign(
            Type=[key],
            predictor=[pm],
            TP=[TP],
            FP=[FP],
            TN=[TN],
            FN=[FN],
            total=[total],
            TPR=[TPR],
            TNR=[TNR],
            precision=[precision],
            Fscore=[2 * (precision * TPR) / (precision + TPR)],
            accuracy=[(TP + TN) / total],
            kappa=[metrics.cohen_kappa_score(pos, gtbe)],
            profit=[profit],
            index=[0],
            ROME=[profit / (cost * (TP + FP))],
            contact=[contact],
            AUC=[metrics.auc(fpr, tpr)],
        )

    result = pd.DataFrame()
    for key, val in dct.items():
        for p in pred:
            result = pd.concat([result, calculate_metrics(key, val, p)], axis=0)

    result.index = range(result.shape[0])
    result["index"] = result.groupby("Type").profit.transform(lambda x: x / x.max())
    return result.round(dec)


def auc(rvar, pred, lev=1):
    """
    Calculate area under the RO curve (AUC)

    Calculation adapted from https://stackoverflow.com/a/50202118/1974918

    Parameters
    ----------
    rvar : Pandas series or numpy vector
        Vector with the response variable
    pred : Pandas series or numpy vector
        Vector with model predictions
    lev : str
        Name of the 'success' level in rvar

    Returns
    -------
    float :
        AUC metric

    Examples
    --------
    auc(dvd.buy, np.random.uniform(size=20000), "yes")
    auc(dvd.buy, rsm.ifelse(dvd.buy == "yes", 1, 0), "yes")
    """
    if type(rvar[0]) != bool or lev is not None:
        rvar = rvar == lev

    n1 = np.sum(np.logical_not(rvar))
    n2 = np.sum(rvar)

    U = np.sum(rankdata(pred)[np.logical_not(rvar)]) - n1 * (n1 + 1) / 2
    wt = U / n1 / n2
    return ifelse(wt < 0.5, 1 - wt, wt)
