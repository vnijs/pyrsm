import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit
from pyrsm.utils import ifelse
from pyrsm.stats import weighted_mean, weighted_sd
from pyrsm.perf import auc


def sig_stars(pval):
    pval = np.nan_to_num(pval, nan=1.0)
    cutpoints = np.array([0.001, 0.01, 0.05, 0.1, np.Inf])
    symbols = np.array(["***", "**", "*", ".", " "])
    return [symbols[p < cutpoints][0] for p in pval]


def or_ci(fitted, alpha=0.05, intercept=False, importance=False, data=None, dec=3):
    """
    Confidence interval for Odds ratios

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in output (True or False)
    importance : int
        Calculate variable importance. Only meaningful if data
        used in estimation was standardized prior to model
        estimation
    data : Pandas dataframe
        Unstandardized data used to calculate descriptive
        statistics
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe with Odd-ratios and confidence intervals
    """

    df = pd.DataFrame(np.exp(fitted.params), columns=["OR"]).dropna()
    df["OR%"] = 100 * ifelse(df["OR"] < 1, -(1 - df["OR"]), df["OR"] - 1)

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    df[[f"{low}%", f"{high}%"]] = np.exp(fitted.conf_int(alpha=alpha))
    df["p.values"] = ifelse(fitted.pvalues < 0.001, "< .001", fitted.pvalues.round(dec))
    df["  "] = sig_stars(fitted.pvalues)
    df["OR%"] = [f"{round(o, max(dec-2, 0))}%" for o in df["OR%"]]
    df = df.reset_index()

    if importance:
        df["dummy"] = df["index"].str.contains("[T", regex=False)
        df["importance"] = (
            pd.DataFrame().assign(OR=df["OR"], ORinv=1 / df["OR"]).max(axis=1)
        )

    if isinstance(data, pd.DataFrame):
        # using a fake response variable variable
        data = data.assign(__rvar__=1).copy()
        form = "__rvar__ ~ " + fitted.model.formula.split("~", 1)[1]
        exog = pd.DataFrame(smf.logit(formula=form, data=data).exog)
        weights = fitted._freq_weights
        if sum(weights) > len(weights):

            def wmean(x):
                return weighted_mean(x, weights)

            def wstd(x):
                return weighted_sd(pd.DataFrame(x), weights)[0]

            df = pd.concat(
                [df, exog.apply([wmean, wstd, "min", "max"]).T],
                axis=1,
            )
        else:
            df = pd.concat([df, exog.apply(["mean", "std", "min", "max"]).T], axis=1)

    if intercept is False:
        df = df.loc[df["index"] != "Intercept"]

    return df.round(dec)


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

    # iloc to reverse order
    df = or_ci(fitted, alpha=alpha, intercept=intercept, dec=100).dropna().iloc[::-1]

    if incl is not None:
        incl = ifelse(isinstance(incl, list), incl, [incl])
        rx = "(" + "|".join([f"^{v}$|^{v}\\[" for v in incl]) + ")"
        incl = df["index"].str.match(rf"{rx}")
        if intercept:
            incl[0] = True
        df = df[incl]

    if excl is not None:
        excl = ifelse(isinstance(excl, list), excl, [excl])
        rx = "(" + "|".join([f"^{v}$|^{v}\\[" for v in excl]) + ")"
        excl = df["index"].str.match(rf"{rx}")
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


def vif(fitted, dec=3):
    """
    Calculate the Variance Inflation Factor (VIF) associated with each
    exogenous variable

    Status
    ------
    WIP port of VIF calculation from R's car:::vif.default to Python

    Parameters
    ----------
    fitted : A fitted (logistic) regression model
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe sorted by VIF score
    """

    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        # legacy for when only an un-fitted model was accepted
        model = fitted

    vif = [variance_inflation_factor(model.exog, i) for i in range(model.exog.shape[1])]
    df = pd.DataFrame(model.exog_names, columns=["variable"])
    df["vif"] = vif
    df["Rsq"] = 1 - 1 / df["vif"]

    if "Intercept" in model.exog_names:
        df = df[df["variable"] != "Intercept"]

    df = df.sort_values("vif", ascending=False).reset_index(drop=True)

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

    # generate predictions
    prediction = fitted.predict(df)

    # set up the data in df in the same was as the exog data
    # that is part of fitted.model.exog
    # use a fake response variable
    df = df.assign(__rvar__=1).copy()
    form = "__rvar__ ~ " + fitted.model.formula.split("~", 1)[1]
    exog = smf.logit(formula=form, data=df).exog

    low, high = [alpha / 2, 1 - (alpha / 2)]
    Xb = np.dot(exog, fitted.params)
    se = np.sqrt((exog.dot(fitted.cov_params()) * exog).sum(-1))
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
        AUC=[auc(fitted.model.endog, fitted.fittedvalues)],
        log_likelihood=fitted.llf,
        BIC=[fitted.bic_llf],
        AIC=[fitted.aic],
        chisq=[fitted.pearson_chi2],
        chisq_df=[fitted.df_model],
        chisq_pval=[1 - stats.chi2.cdf(fitted.pearson_chi2, fitted.df_model)],
        nobs=[fitted.nobs],
    )

    output = f"""
Pseudo R-squared (McFadden): {mfit.pseudo_rsq_mcf.values[0].round(dec)}
Pseudo R-squared (McFadden adjusted): {mfit.pseudo_rsq_mcf_adj.values[0].round(dec)}
Area under the RO Curve (AUC): {mfit.AUC.values[0].round(dec)}
Log-likelihood: {mfit.log_likelihood.values[0].round(dec)}, AIC: {mfit.AIC.values[0].round(dec)}, BIC: {mfit.BIC.values[0].round(dec)}
Chi-squared: {mfit.chisq.values[0].round(dec)} df({mfit.chisq_df.values[0]}), p.value {np.where(mfit.chisq_pval.values[0] < .001, "< 0.001", mfit.chisq_pval.values[0].round(dec))} 
Nr obs: {mfit.nobs.values[0]:,}
"""

    if prn:
        print(output)
    else:
        return mfit
