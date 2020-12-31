import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from scipy.stats import norm
from pyrsm.utils import ifelse


def or_ci(fitted, alpha=0.05, intercept=False, dec=3):
    """
    Confidence interval for Odds ratios

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    dec : int
        Number of decimal places

    Returns
    -------
    Pandas dataframe with Odd-ratios and confidence intervals
    """

    df = pd.DataFrame(np.exp(fitted.params), columns=["OR"])
    df["OR%"] = 100 * ifelse(df["OR"] < 1, -(1 - df["OR"]), df["OR"] - 1)

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    df[[f"{low}%", f"{high}%"]] = np.exp(fitted.conf_int(alpha=alpha))

    if dec is not None:
        df = df.round(dec)

    df["OR%"] = [f"{OR}%" for OR in df["OR%"]]
    df = df.reset_index()

    if intercept is False:
        df = df.loc[df["index"] != "Intercept"]

    return df


def or_conf_int(fitted, alpha=0.05, intercept=False, dec=3):
    """ Shortcut to or_ci """
    return or_ci(fitted, alpha=0.05, intercept=False, dec=3)


def or_plot(fitted, alpha=0.05, intercept=False):
    """
    Odds ratio plot

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in plot (True or False)

    Returns
    -------
    Matplotlit object
        Plot of Odds ratios
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

    Status
    ------
    WIP port the VIF calculation from R's car:::vif.default to Python

    Parameters
    ----------
    model :  A specified model that has not yet been fitted
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe sorted by VIF score
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


def predict_ci(fitted, df, alpha=0.05):
    """
    Compute predicted probabilities with confidence intervals based on a
    logistic regression model

    Parameters
    ----------
    fitted : Logistic regression model fitted using the statsmodels formula interface
    df : Pandas dataframe with input data for prediction
    alpha : float
        Significance level (0-1). Default is 0.05

    Returns
    -------
    Pandas dataframe with probability predictions and lower and upper confidence bounds

    Example
    -------
    import numpy as np
    import statsmodels.formula.api as smf
    import pandas as pd

    # simulate data
    np.random.seed(1)
    x1 = np.arange(100)
    x2 = pd.Series(["a", "b", "c", "a"], dtype="category").sample(100, replace=True)
    y = (x1 * 0.5 + np.random.normal(size=100, scale=10) > 30).astype(int)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    # estimate the model
    model = smf.logit(formula="y ~ x1 + x2", data=df).fit()
    model.summary()
    pred = predict_ci(model, df)

    plt.clf()
    plt.plot(x1, pred["prediction"])
    plt.plot(x1, pred["2.5%"], color='black', linestyle="--", linewidth=0.5)
    plt.plot(x1, pred["97.5%"], color='black', linestyle="--", linewidth=0.5)
    plt.show()
    """

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be a numeric value between 0 and 1")

    # generate prediction
    prediction = fitted.predict(df)

    # adding a fake endogenous variable
    df = df.copy()  # making a full copy
    df["__endog__"] = 1
    form = "__endog__ ~ " + fitted.model.formula.split("~", 1)[1]
    df = smf.logit(formula=form, data=df).exog

    low, high = [alpha / 2, 1 - (alpha / 2)]
    Xb = np.dot(df, fitted.params)
    se = np.sqrt((df.dot(fitted.cov_params()) * df).sum(-1))
    me = norm.ppf(high) * se
    lb = np.exp(Xb - me)
    ub = np.exp(Xb + me)

    return pd.DataFrame(
        {
            "prediction": prediction,
            f"{low*100}%": lb / (1 + lb),
            f"{high*100}%": ub / (1 + ub),
        }
    )
