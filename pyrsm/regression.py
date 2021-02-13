import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from pyrsm.utils import ifelse
from pyrsm.logit import sig_stars


def coef_plot(fitted, alpha=0.05, intercept=False, incl=None, excl=None, figsize=None):
    """
    Coefficient plot

    Parameters
    ----------
    fitted : A fitted linear regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in coefficient plot (True or False)
    incl : str or list of strings
        Variables to include in the coefficient plot. All will be included by default
    excl : str or list of strings
        Variables to exclude from the coefficient plot. None are excluded by default

    Returns
    -------
    Matplotlit object
        Plot of Odds ratios
    """
    df = fitted.conf_int(alpha=alpha).reset_index().iloc[::-1]
    df["coefficient"] = fitted.params[df["index"]].values

    if not intercept:
        df = df.query('index != "Intercept"')

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
    df.columns = ["index", f"{low}%", f"{high}%", "coefficient"]
    err = [df["coefficient"] - df[f"{low}%"], df[f"{high}%"] - df["coefficient"]]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.axvline(0, ls="dashdot")
    ax.errorbar(x="coefficient", y="index", data=df, xerr=err, fmt="none")
    ax.scatter(x="coefficient", y="index", data=df)
    ax.set(xlabel="Coefficient")
    return ax


def coef_ci(fitted, alpha=0.05, intercept=False, dec=3):
    """
    Confidence interval for coefficient from linear regression

    Parameters
    ----------
    fitted : A fitted linear regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in the output (True or False)
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe with regression coefficients and confidence intervals
    """

    df = pd.DataFrame({"coefficient": fitted.params})

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    df[[f"{low}%", f"{high}%"]] = fitted.conf_int(alpha=alpha)

    if dec is None:
        df["p.values"] = ifelse(fitted.pvalues < 0.001, "< .001", fitted.pvalues)
    else:
        df = df.round(dec)
        df["p.values"] = ifelse(
            fitted.pvalues < 0.001, "< .001", fitted.pvalues.round(dec)
        )

    df["  "] = sig_stars(fitted.pvalues)
    df = df.reset_index()

    if intercept is False:
        df = df.loc[df["index"] != "Intercept"]

    return df


# def predict_ci(fitted, df, alpha=0.05):
#     """
#     Compute predicted probabilities with confidence intervals based on a
#     linear regression model

#     Parameters
#     ----------
#     fitted : Linear regression model fitted using the statsmodels formula interface
#     df : Pandas dataframe with input data for prediction
#     alpha : float
#         Significance level (0-1). Default is 0.05

#     Returns
#     -------
#     Pandas dataframe with predictions and lower and upper confidence bounds
#     """

#     if alpha < 0 or alpha > 1:
#         raise ValueError("alpha must be a numeric value between 0 and 1")

#     # generate prediction
#     prediction = fitted.predict(df)

#     # adding a fake endogenous variable
#     df = df.copy()  # making a full copy
#     df["__endog__"] = 1
#     form = "__endog__ ~ " + fitted.model.formula.split("~", 1)[1]
#     df = smf.ols(formula=form, data=df).exog

#     low, high = [alpha / 2, 1 - (alpha / 2)]
#     Xb = np.dot(df, fitted.params)
#     se = np.sqrt((df.dot(fitted.cov_params()) * df).sum(-1))
#     me = stats.norm.ppf(high) * se

#     return pd.DataFrame(
#         {
#             "prediction": prediction,
#             f"{low*100}%": Xb - me,
#             f"{high*100}%": Xb + me,
#         }
#     )
