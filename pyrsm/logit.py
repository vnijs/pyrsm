import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit
from pyrsm.utils import ifelse


def sig_stars(pval):
    cutpoints = np.array([0.001, 0.01, 0.05, 0.1, 1])
    symbols = np.array(["***", "**", "*", ".", " "])
    return [symbols[p < cutpoints][0] for p in pval]


def or_ci(fitted, alpha=0.05, intercept=False, dec=3):
    """
    Confidence interval for Odds ratios

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in output (True or False)
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe with Odd-ratios and confidence intervals
    """

    df = pd.DataFrame(np.exp(fitted.params), columns=["OR"])
    df["OR%"] = 100 * ifelse(df["OR"] < 1, -(1 - df["OR"]), df["OR"] - 1)

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    df[[f"{low}%", f"{high}%"]] = np.exp(fitted.conf_int(alpha=alpha))

    if dec is None:
        df["p.values"] = ifelse(fitted.pvalues < 0.001, "< .001", fitted.pvalues)
    else:
        df = df.round(dec)
        df["p.values"] = ifelse(
            fitted.pvalues < 0.001, "< .001", fitted.pvalues.round(dec)
        )

    df["  "] = sig_stars(fitted.pvalues)
    df["OR%"] = [f"{OR}%" for OR in df["OR%"]]
    df = df.reset_index()

    if intercept is False:
        df = df.loc[df["index"] != "Intercept"]

    return df


def or_conf_int(fitted, alpha=0.05, intercept=False, dec=3):
    """ Shortcut to or_ci """
    return or_ci(fitted, alpha=0.05, intercept=False, dec=dec)


def or_plot(fitted, alpha=0.05, intercept=False, incl=None, excl=None, figsize=None):
    """
    Odds ratio plot

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in odds-ratio plot (True or False)
    incl : str or list of strings
        Variables to include in the odds-ratio plot. All will be included by default
    excl : str or list of strings
        Variables to exclude from the odds-ratio plot. None are excluded by default

    Returns
    -------
    Matplotlit object
        Plot of Odds ratios
    """

    df = or_ci(fitted, alpha=alpha, intercept=intercept, dec=None).iloc[::-1]

    if incl is not None:
        incl = ifelse(isinstance(incl, list), incl, [incl])
        rx = "(" + "|".join([f"^\b{v}|^{v}\\[" for v in incl]) + ")"
        incl = df["index"].str.match(fr"{rx}")
        if intercept:
            incl[0] = True
        df = df[incl]

    if excl is not None:
        excl = ifelse(isinstance(excl, list), excl, [excl])
        rx = "(" + "|".join([f"^\b{v}|^{v}\\[" for v in excl]) + ")"
        excl = df["index"].str.match(fr"{rx}")
        if intercept:
            excl[0] = False
        df = df[~excl]

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    err = [df["OR"] - df[f"{low}%"], df[f"{high}%"] - df["OR"]]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
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
    me = stats.norm.ppf(high) * se

    return pd.DataFrame(
        {
            "prediction": prediction,
            f"{low*100}%": expit(Xb - me),
            f"{high*100}%": expit(Xb + me),
        }
    )


def model_fit(fitted, dec=3, prn=True):
    """
    Compute various model fit statistics for a fitted logistic regression model

    Parameters
    ----------
    fitted : statmodels glm object
        Logistic regression model fitted using statsmodels
    dec : int
        Number of decimal places to use in rounding
    prn : bool
        If True, print output, else return a Pandas dataframe with the results

    Returns
    -------
        If prn is True, print output, else return a Pandas dataframe with the results
    """

    mfit = pd.DataFrame().assign(
        pseudo_rsq_mcf=[1 - fitted.llf / fitted.llnull],
        pseudo_rsq_mcf_adj=[1 - (fitted.llf - fitted.df_model) / fitted.llnull],
        log_likelihood=fitted.llf,
        BIC=[fitted.bic_llf],
        AIC=[fitted.aic],
        chisq=[fitted.pearson_chi2],
        chisq_df=[fitted.df_model],
        chisq_pval=[1 - stats.chi2.cdf(fitted.pearson_chi2, fitted.df_model)],
        nobs=[fitted.nobs],
    )

    output = f"""
Pseudo R-squared (McFadden): {mfit["pseudo_rsq_mcf"].values[0].round(dec)}
Pseudo R-squared (McFadden adjusted): {mfit["pseudo_rsq_mcf_adj"].values[0].round(dec)}
Log-likelihood: {mfit["log_likelihood"].values[0].round(dec)}, AIC: {mfit["AIC"].values[0].round(dec)}, BIC: {mfit["BIC"].values[0].round(dec)}
Chi-squared: {mfit["chisq"].values[0].round(dec)} df({mfit["chisq_df"].values[0]}), p.value {np.where(mfit["chisq_pval"].values[0] < .001, "< 0.001", mfit["chisq_pval"].values[0].round(dec))} 
Nr obs: {mfit["nobs"].values[0]:,}
"""

    if prn:
        print(output)
    else:
        return mfit
