import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyrsm.utils import ifelse
from pyrsm.logit import sig_stars
from pyrsm.utils import expand_grid
from sklearn import metrics
import seaborn as sns
from scipy import stats
import statsmodels.api as sm


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


def evalreg(df, rvar, pred, dec=3):
    """
    Evaluate regression models. Calculates R-squared, MSE, and MAE

    Parameters
    ----------
    df : Pandas dataframe or a dictionary of dataframes with keys to show results for
        multiple model predictions and datasets (training and test)
    rvar : str
        Name of the response variable column in df
    pred : str
        Name of the column, of list of column names, in df with model predictions
    dec : int
        Number of decimal places to use in rounding

    Examples
    --------
    """

    dct = ifelse(type(df) is dict, df, {"All": df})
    pred = ifelse(type(pred) is list, pred, [pred])

    def calculate_metrics(key, dfm, pm):
        return pd.DataFrame().assign(
            Type=[key],
            predictor=[pm],
            n=[dfm.shape[0]],
            r2=[metrics.r2_score(dfm[rvar], dfm[pm])],
            mse=[metrics.mean_squared_error(dfm[rvar], dfm[pm])],
            mae=[metrics.mean_absolute_error(dfm[rvar], dfm[pm])],
        )

    result = pd.DataFrame()
    for key, val in dct.items():
        for p in pred:
            result = result.append(calculate_metrics(key, val, p))

    result.index = range(result.shape[0])
    return result.round(dec)


def reg_dashboard(reg, rvar, nobs=1000):
    """
    Plot regression residual dashboard

    Parameters
    ----------
    reg : Object with fittedvalues and residuals
    rvar : Model response variable as a pandas series
    nobs: int
        Number of observerations to use for plots.
        Set to None or -1 to plot all values.
        The Residuals vs Order plot will only be valid
        if all observations are plotted
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.25, hspace=0.4)

    data = pd.DataFrame().assign(
        fitted=reg.fittedvalues,
        actual=rvar,
        resid=reg.resid,
        std_resid=reg.resid / np.std(reg.resid),
        order=np.arange(len(rvar)),
    )

    if (nobs != -1 and nobs is not None) and (nobs < data.shape[0]):
        data = data.sample(nobs)

    sns.regplot(x="fitted", y="actual", data=data, ax=axes[0, 0]).set(
        title="Actual vs Fitted values", xlabel="Fitted values", ylabel="Actual values"
    )
    sns.regplot(x="fitted", y="resid", data=data, ax=axes[0, 1]).set(
        title="Residuals vs Fitted values", xlabel="Fitted values", ylabel="Residuals"
    )
    sns.lineplot(x="order", y="resid", data=data, ax=axes[1, 0]).set(
        title="Residuals vs Row order", ylabel="Residuals", xlabel=None
    )
    sm.qqplot(data.resid, line="s", ax=axes[1, 1])
    axes[1, 1].title.set_text("Normal Q-Q plot")
    pdp = data.resid.plot.hist(
        ax=axes[2, 0],
        title="Histogram of residuals",
        xlabel="Residuals",
        rot=0,
        color="slateblue",
    )
    pdp.set_xlabel("Residuals")
    sns.kdeplot(
        data.std_resid, color="green", shade=True, ax=axes[2, 1], common_norm=True
    )

    # from https://stackoverflow.com/a/52925509/1974918
    norm_x = np.arange(-3, +3, 0.01)
    norm_y = stats.norm.pdf(norm_x)
    sns.lineplot(x=norm_x, y=norm_y, lw=1, ax=axes[2, 1]).set(
        title="Residuals vs Normal density", xlabel="Residuals"
    )


def sim_prediction(df, vary=[]):
    """
    Simulate data for prediction

    Parameters
    ----------
    df : Pandas dataframe
    vary : List of column names of Dictionary with keys and values to use

    Returns:
    ----------
    Pandas dataframe with values to use for estimation
    """

    def fix_value(s):
        if pd.api.types.is_numeric_dtype(s.dtype):
            return s.mean()
        else:
            return s.value_counts().idxmax()

    dct = {c: [fix_value(df[c])] for c in df.columns}
    if type(vary) is dict:
        # user provided values and ranges
        for key, val in vary.items():
            dct[key] = val
    else:
        # auto derived values and ranges
        for v in vary:
            if pd.api.types.is_numeric_dtype(df[v].dtype):
                if df[v].nunique() > 2:
                    dct[v] = np.linspace(df[v].min(), df[v].max(), 5)
                else:
                    dct[v] = [df[v].min(), df[v].max()]
            else:
                dct[v] = df[v].unique()

    return expand_grid(dct)
