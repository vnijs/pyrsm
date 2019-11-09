import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.stats.outliers_influence import variance_inflation_factor


def or_conf_int(fitted, alpha=0.05, intercept=False, dec=3):
    """
    Confidence interval for Odds ratios

    Arguments:
    fitted   A fitted logistic regression model
    alpha   Significance level
    dec     Number of decimal places

    Return:
    A dataframe with the Odd-ratios and confidence interval
    """
    df = pd.DataFrame(np.exp(fitted.params), columns=["OR"])
    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    df[[f"{low}%", f"{high}%"]] = np.exp(fitted.conf_int(alpha=alpha))

    if dec is not None:
        df = df.round(dec)

    df = df.reset_index()

    if intercept is False:
        df = df.loc[df["index"] != "Intercept"]

    return df


def or_plot(fitted, alpha=0.05, intercept=False):
    """
    Odds ratio plot

    Arguments:
    fitted       A fitted logistic regression fitted
    alpha       Significance level
    intercept   Include intercept in plot True or False
    """
    df = or_conf_int(fitted, alpha=alpha, intercept=intercept, dec=None).iloc[::-1]

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    err = [df["OR"] - df[f"{low}%"], df[f"{high}%"] - df["OR"]]

    fig, ax = plt.subplots()
    ax.axvline(1, ls="dashdot")
    ax.errorbar(x="OR", y="index", data=df, xerr=err, fmt="none")
    ax.scatter(x="OR", y="index", data=df)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_locator(ticker.LogLocator(subs=[0.1, 0.2, 0.5, 1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.set(xlabel="Odds-ratio")
    return ax


def vif(model, dec=3):
    """
    Calculate the Variance Inflation Factor (VIF) associated with each
    exogenous variable

    WIP port the VIF calculation from R's car:::vif.default to Python

    Arguments:
    model   A specified model that has not yet been fit
    dec     Decimal places to use in rounding

    Return:
    A dataframe sorted by VIF score
    """
    vif = [variance_inflation_factor(model.exog, i) for i in range(model.exog.shape[1])]
    df = pd.DataFrame(model.exog_names, columns=["variable"])
    df["vif"] = vif
    df["Rsq"] = 1 - 1 / df["vif"]

    if "Intercept" in model.exog_names:
        df = df.loc[df["variable"] != "Intercept"]

    df = df.sort_values("vif", ascending=False)
    df.index = range(df.shape[0])

    if dec is not None:
        df = df.round(dec)

    return df
