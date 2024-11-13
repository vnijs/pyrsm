from typing import Union, Optional
import re
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels as sm
import statsmodels.api as sma
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from math import ceil
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.special import expit
from pyrsm.utils import ifelse, expand_grid, check_dataframe
from pyrsm.stats import weighted_mean, weighted_sd
from .perf import auc


def make_train(
    data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
    strat_var: str | list[str] = None,
    test_size: float = 0.2,
    random_state: int = 1234,
):
    """
    Use stratified sampling on one or more variables to create a training
    variable for your dataset
    """
    splits = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    data = check_dataframe(data)
    training_var = np.zeros(data.shape[0])
    for train_index, test_index in splits.split(data, data[strat_var]):
        training_var[train_index] = 1
    return training_var


def extract_evars(model, cn):
    """
    Extract a list of the names of the explanatory variables in a statsmodels model
    """
    pattern = r"\b\w+\b"
    evars = re.findall(pattern, model.formula)[1:]
    evars = [v for v in evars if v in cn]
    return [v for i, v in enumerate(evars) if v not in evars[:i]]


def extract_rvar(model, cn):
    """
    Extract name of the response variable in a statsmodels model
    """
    pattern = r"\b\w+\b"
    return re.findall(pattern, model.formula)[0]


def convert_to_list(v):
    if isinstance(v, list) or v is None:
        return v
    elif isinstance(v, str):
        return [v]
    else:
        return list(v)


def convert_binary(data, rvar, lev):
    cb = ifelse(data[rvar] == lev, 1, ifelse(data[rvar].isna(), np.nan, 0))
    data = data.drop(columns=rvar).copy()
    data.loc[:, rvar] = cb
    return data


def conditional_get_dummies(df):
    for column in df.select_dtypes(include=["object", "category"]).columns:
        unique_values = df[column].nunique()
        if unique_values == 2:  # Binary variable
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
        else:  # Multi-level variable
            dummies = pd.get_dummies(df[column], prefix=column)
        # Drop the original column and concatenate the new dummy columns
        df = pd.concat([df.drop(column, axis=1), dummies], axis=1)
    return df


def sig_stars(pval):
    pval = np.nan_to_num(pval, nan=1.0)
    cutpoints = np.array([0.001, 0.01, 0.05, 0.1, np.inf])
    symbols = np.array(["***", "**", "*", ".", " "])
    return [symbols[p < cutpoints][0] for p in pval]


def get_mfit(fitted) -> tuple[Optional[dict], Optional[str]]:
    """
    Get model fit statistics

    Parameters
    ----------
    fitted : A fitted linear or logistic 4egression model
    """

    mfit_dct = None
    model_type = None
    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        fw = None
        if fitted.model._has_freq_weights:
            fw = fitted.model.freq_weights

        # gets same results as in R
        lrtest = -2 * (fitted.llnull - fitted.llf)
        mfit_dct = {
            "pseudo_rsq_mcf": [1 - fitted.llf / fitted.llnull],
            "pseudo_rsq_mcf_adj": [1 - (fitted.llf - fitted.df_model) / fitted.llnull],
            "AUC": [auc(fitted.model.endog, fitted.fittedvalues, weights=fw)],
            "log_likelihood": fitted.llf,
            "AIC": [fitted.aic],
            "BIC": [fitted.bic_llf],
            # "chisq": [fitted.pearson_chi2],
            "chisq": [lrtest],
            "chisq_df": [fitted.df_model],
            # "chisq_pval": [1 - stats.chi2.cdf(fitted.pearson_chi2, fitted.df_model)],
            "chisq_pval": [stats.chi2.sf(lrtest, fitted.df_model)],
            "nobs": [fitted.nobs],
        }

        model_type = "logistic"

    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        mfit_dct = {
            "rsq": [fitted.rsquared],
            "rsq_adj": [fitted.rsquared_adj],
            "fvalue": [fitted.fvalue],
            "ftest_df_model": [fitted.df_model],
            "ftest_df_resid": [fitted.df_resid],
            "ftest_pval": [fitted.f_pvalue],
            "nobs": [fitted.nobs],
        }

        model_type = "regression"

    return mfit_dct, model_type


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
        incl = ifelse(isinstance(incl, str), [incl], incl)
        rx = "(" + "|".join([f"^{v}$|^{v}\\[" for v in incl]) + ")"
        incl = df["index"].str.match(rf"{rx}")
        if intercept:
            incl[0] = True
        df = df[incl]

    if excl is not None:
        excl = ifelse(isinstance(excl, str), [excl], excl)
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

    is_categorical = {
        item.split("[T.")[0]: ("[T." in item) for item in model.exog_names
    }
    if sum(is_categorical.values()) > 0:
        evar = list(is_categorical.keys())[1:]
        exog = model.exog[:, 1:]
        Xcorr = np.linalg.det(np.corrcoef(exog, rowvar=False))
        df = pd.DataFrame(exog, columns=model.exog_names[1:])
        vif = []
        for col, cat in is_categorical.items():
            if col == "Intercept":
                continue
            elif cat:
                select = [f"{col}[T." in c for c in df.columns]
                Vcorr = np.linalg.det(df.loc[:, select].corr().values)
                drop = [f"{col}[T." not in c for c in df.columns]
                Ocorr = np.linalg.det(df.loc[:, drop].corr().values)
            else:
                Vcorr = 1
                Ocorr = np.linalg.det(df.drop(col, axis=1).corr().values)

            vif.append(Vcorr * Ocorr / Xcorr)
        df = pd.DataFrame(evar, columns=["index"])
    else:
        vif = [
            variance_inflation_factor(model.exog, i) for i in range(model.exog.shape[1])
        ]
        df = pd.DataFrame(model.exog_names, columns=["index"])

    df["vif"] = vif
    df["Rsq"] = 1 - 1 / df["vif"]

    if "Intercept" in model.exog_names:
        df = df[df["index"] != "Intercept"]

    df = df.sort_values("vif", ascending=False).set_index("index")
    df.index.name = None

    if dec is not None:
        df = df.round(dec)

    return df


def predict_ci(fitted, df, conf=0.95, alpha=None):
    """
    Compute predicted probabilities with confidence intervals based on a
    logistic regression model

    Parameters
    ----------
    fitted : Logistic regression model fitted using the statsmodels formula interface
    df : Pandas dataframe with input data for prediction
    conf : float
        Confidence level (0-1). Default is 0.95
    alpha : float
        Significance level (0-1). Default is 0.05

    Returns
    -------
    Pandas DataFrame with probability predictions and lower and upper confidence bounds

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

    if conf < 0 or conf > 1:
        raise ValueError(
            "Confidence level (conf) must be a numeric value between 0 and 1"
        )

    if alpha is not None:
        raise ValueError(
            "The alpha argument has been deprecated. Use the confidence level (conf) instead (1-alpha)."
        )

    # generate predictions
    prediction = fitted.predict(df)

    # set up the data in df in the same was as the exog data
    # that is part of fitted.model.exog
    # use a fake response variable
    df = df.assign(__rvar__=1).copy()
    form = "__rvar__ ~ " + fitted.model.formula.split("~", 1)[1]
    exog = smf.logit(formula=form, data=df).exog

    low, high = [(1.0 - conf) / 2.0, (1.0 - (1.0 - conf) / 2.0)]
    Xb = np.dot(exog, fitted.params)
    se = np.sqrt((exog.dot(fitted.cov_params()) * exog).sum(-1))
    me = stats.norm.ppf(high) * se

    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        return pd.DataFrame(
            {
                "prediction": prediction,
                f"{low*100:.2f}%": expit(Xb - me),
                f"{high*100:.2f}%": expit(Xb + me),
            }
        )
    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        return pd.DataFrame(
            {
                "prediction": prediction,
                f"{low*100:.2f}%": Xb - me,
                f"{high*100:.2f}%": Xb + me,
            }
        )


def model_fit(fitted, dec: int = 3, prn: bool = True) -> Union[str, pd.DataFrame]:
    """
    Compute various model fit statistics for a fitted linear or logistic regression model

    Parameters
    ----------
    fitted : statmodels ols or glm object
        Regression model fitted using statsmodels
    dec : int
        Number of decimal places to use in rounding
    prn : bool
        If True, print output, else return a Pandas dataframe with the results

    Returns
    -------
        If prn is True, print output, else return a Pandas dataframe with the results
    """
    if hasattr(fitted, "df_resid"):
        weighted_nobs = (
            fitted.df_resid + fitted.df_model + int(hasattr(fitted.params, "Intercept"))
        )
        if weighted_nobs > fitted.nobs:
            fitted.nobs = weighted_nobs
    mfit_dct, model_type = get_mfit(fitted)
    if not mfit_dct:
        return "Only linear and logistic regression models are currently supported."

    mfit = pd.DataFrame(mfit_dct)

    if prn:
        if model_type == "logistic":
            output = f"""Pseudo R-squared (McFadden): {mfit.pseudo_rsq_mcf.values[0].round(dec)}
Pseudo R-squared (McFadden adjusted): {mfit.pseudo_rsq_mcf_adj.values[0].round(dec)}
Area under the RO Curve (AUC): {mfit.AUC.values[0].round(dec)}
Log-likelihood: {mfit.log_likelihood.values[0].round(dec)}, AIC: {mfit.AIC.values[0].round(dec)}, BIC: {mfit.BIC.values[0].round(dec)}
Chi-squared: {mfit.chisq.values[0].round(dec)}, df({mfit.chisq_df.values[0]}), p.value {np.where(mfit.chisq_pval.values[0] < .001, "< 0.001", mfit.chisq_pval.values[0].round(dec))} 
Nr obs: {mfit.nobs.values[0]:,.0f}"""
        elif model_type == "regression":
            output = f"""R-squared: {mfit.rsq.values[0].round(dec)}, Adjusted R-squared: {mfit.rsq_adj.values[0].round(dec)}
F-statistic: {mfit.fvalue[0].round(dec)} df({mfit.ftest_df_model.values[0]:.0f}, {mfit.ftest_df_resid.values[0]:.0f}), p.value {np.where(mfit.ftest_pval.values[0] < .001, "< 0.001", mfit.ftest_pval.values[0].round(dec))}
Nr obs: {mfit.nobs.values[0]:,.0f}"""
        else:
            output = "Model type not supported"
        return output
    else:
        return mfit


def coef_plot(
    fitted,
    alpha: float = 0.05,
    intercept: bool = False,
    incl: str = None,
    excl: list = [],
    figsize: tuple = None,
):
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
    Matplotlib object
        Plot of Odds ratios
    """
    df = fitted.conf_int(alpha=alpha).reset_index().iloc[::-1]
    df["coefficient"] = fitted.params[df["index"]].dropna().values

    if not intercept:
        df = df.query('index != "Intercept"')

    if incl is not None:
        incl = ifelse(isinstance(incl, str), [incl], incl)
        rx = "(" + "|".join([f"^\b{v}|^{v}\\[" for v in incl]) + ")"
        incl = df["index"].str.match(rf"{rx}")
        if intercept:
            incl[0] = True
        df = df[incl]

    if len(excl) > 0 and excl is not None:
        excl = ifelse(isinstance(excl, str), [excl], excl)
        rx = "(" + "|".join([f"^\b{v}|^{v}\\[" for v in excl]) + ")"
        excl = df["index"].str.match(rf"{rx}")
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


def coef_ci(fitted, alpha: float = 0.05, intercept: bool = True, dec: int = 3):
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
    if intercept is False:
        df = df[df.index != "Intercept"]

    return df


def evalreg(df, rvar: str, pred: str, dec: int = 3):
    """
    Evaluate regression models. Calculates R-squared, MSE, and MAE

    Parameters
    ----------
    df : Pandas DataFrame or a dictionary of DataFrames with keys to show results for
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

    dct = ifelse(isinstance(df, dict), df, {"All": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)

    def calculate_metrics(key, dfm, pm):
        return pd.DataFrame().assign(
            Type=[key],
            predictor=[pm],
            n=[dfm.shape[0]],
            r2=[metrics.r2_score(dfm[rvar], dfm[pm])],
            mse=[metrics.mean_squared_error(dfm[rvar], dfm[pm])],
            mae=[metrics.mean_absolute_error(dfm[rvar], dfm[pm])],
        )

    result = pd.concat(
        [calculate_metrics(key, val, p) for key, val in dct.items() for p in pred],
        axis=0,
    )
    result.index = range(result.shape[0])
    return result.round(dec)


def reg_dashboard(fitted, nobs: int = 1000):
    """
    Plot regression residual dashboard

    Parameters
    ----------
    fitted : Object with fitted values and residuals
    nobs: int
        Number of observations to use for plots.
        Set to None or -1 to plot all values.
        The Residuals vs Order plot will only be valid
        if all observations are plotted
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.25, hspace=0.4)

    data = pd.DataFrame().assign(
        fitted=fitted.fittedvalues,
        actual=fitted.model.endog,
        resid=fitted.resid,
        std_resid=fitted.resid / np.std(fitted.resid),
        order=np.arange(fitted.model.endog.shape[0]),
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
    sma.qqplot(data.resid, line="s", ax=axes[1, 1])
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
        data.std_resid, color="green", fill=True, ax=axes[2, 1], common_norm=True
    )

    # from https://stackoverflow.com/a/52925509/1974918
    norm_x = np.arange(-3, +3, 0.01)
    norm_y = stats.norm.pdf(norm_x)
    sns.lineplot(x=norm_x, y=norm_y, lw=1, ax=axes[2, 1]).set(
        title="Residuals vs Normal density", xlabel="Residuals"
    )


def sim_prediction(
    data: pd.DataFrame,
    vary: list = [],
    nnv: int = 5,
    minq: float = 0,
    maxq: float = 1,
) -> pd.DataFrame:
    """
    Simulate data for prediction

    Parameters
    ----------
    data : Pandas DataFrame
    vary : List of column names or Dictionary with keys and values to use
    nnv : int
        Number of values to use to simulate the effect of a numeric variable
    minq : float
        Quantile to use for the minimum value of numeric variables
    maxq : float
        Quantile to use for the maximum value of numeric variables

    Returns:
    ----------
    Pandas DataFrame with values to use for estimation
    """

    def fix_value(s):
        if pd.api.types.is_numeric_dtype(s.dtype):
            return s.mean()
        else:
            return s.value_counts().idxmax()

    dct = {c: [fix_value(data[c])] for c in data.columns}
    dt = data.dtypes
    if isinstance(vary, dict):
        # user provided values and ranges
        for key, val in vary.items():
            dct[key] = val
    else:
        # auto derived values and ranges
        vary = ifelse(isinstance(vary, str), [vary], vary)
        for v in vary:
            if pd.api.types.is_numeric_dtype(data[v].dtype):
                nu = data[v].nunique()
                if nu > 2:
                    dct[v] = np.linspace(
                        np.quantile(data[v], minq),
                        np.quantile(data[v], maxq),
                        min([nu, nnv]),
                    )
                else:
                    dct[v] = [data[v].min(), data[v].max()]
            else:
                dct[v] = data[v].unique()

    return expand_grid(dct, dt)


def scatter_plot(
    fitted, df, nobs: int = 1000, figsize: tuple = None, resid=False
) -> None:
    """
    Scatter plot of explanatory and response variables from a fitted regression

    Parameters
    ----------
    fitted : A fitted linear regression model
    df : Pandas DataFrame
        Data frame with explanatory and response variables
    nobs : int
        Number of observations to use for the scatter plots. The default
        value is 1,000. To use all observations in the plots, use nobs=-1
    figsize : tuple
        A tuple that determines the figure size. If None, size is
        determined based on the number of variables in the model
    resid : bool
        If True, use residuals as the response variable
    """

    exog_names = extract_evars(fitted.model, df.columns)

    if resid:
        endog = fitted.resid
        endog_name = "residuals"
        if df.shape[0] != endog.shape[0]:
            raise ValueError(
                "The number of observations in the fitted model and the data frame must be the same"
            )
        else:
            df = pd.concat(
                [
                    pd.DataFrame({endog_name: endog}),
                    df[exog_names].copy().reset_index(drop=True),
                ],
                axis=1,
            )
    else:
        endog_name = extract_rvar(fitted.model, df.columns)
        df = df[[endog_name] + exog_names].copy()

    nr_plots = len(exog_names)
    if figsize is None:
        figsize = (10, 2 * max(nr_plots, 4))

    fig, ax = plt.subplots(max(ceil(nr_plots / 2), 2), 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    idx = 0

    if nobs < df.shape[0] and nobs != np.inf and nobs != -1:
        df = df.copy().sample(nobs)

    while idx < nr_plots:
        row = idx // 2
        col = idx % 2
        exog_name = exog_names[idx]
        if pd.api.types.is_numeric_dtype(df[exog_name].dtype):
            fig = sns.scatterplot(x=exog_name, y=endog_name, data=df, ax=ax[row, col])
        else:
            fig = sns.stripplot(x=exog_name, y=endog_name, data=df, ax=ax[row, col])
            means = df.groupby(exog_name, observed=False)[endog_name].mean()
            levels = list(means.index)
            # Loop over categories
            for pos, cat in enumerate(levels):
                # Add a line for the mean
                ax[row, col].plot(
                    [pos - 0.5, pos + 0.5],
                    [means.iloc[pos], means.iloc[pos]],
                    color="blue",
                )

        idx += 1

    if nr_plots < 3:
        ax[-1, -1].remove()
        ax[-1, 0].remove()
        if nr_plots == 1:
            ax[0, -1].remove()
    elif nr_plots % 2 == 1:
        ax[-1, -1].remove()


def residual_plot(
    fitted,
    df,
    nobs: int = 1000,
    figsize: tuple = None,
) -> None:
    """
    Plot of variables vs residuals

    Parameters
    ----------
    fitted : A fitted linear regression model
    nobs : int
        Number of observations to use for the scatter plots. The default
        value is 1,000. To use all observations in the plots, use nobs=-1
    figsize : tuple
        A tuple that determines the figure size. If None, size is
        determined based on the number of variables in the model
    """

    scatter_plot(fitted, df, nobs=nobs, figsize=figsize, resid=True)
