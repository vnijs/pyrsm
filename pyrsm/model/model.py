import os
import re
from math import ceil
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import statsmodels as sm
import statsmodels.api as sma
import statsmodels.formula.api as smf
from plotnine import (
    aes,
    coord_flip,
    element_text,
    geom_density,
    geom_errorbarh,
    geom_histogram,
    geom_hline,
    geom_jitter,
    geom_line,
    geom_point,
    geom_segment,
    geom_smooth,
    geom_vline,
    ggplot,
    ggtitle,
    labs,
    scale_x_continuous,
    scale_x_log10,
    scale_y_discrete,
    stat_function,
    stat_qq,
    stat_qq_line,
    theme,
    theme_bw,
)
from scipy import stats
from scipy.special import expit
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, StratifiedShuffleSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pyrsm.notebook import load_state, save_state
from pyrsm.stats import weighted_mean, weighted_sd
from pyrsm.utils import check_dataframe, expand_grid, ifelse

from .perf import auc


def to_pandas_with_categories(pred_data: pl.DataFrame, original_data: pl.DataFrame) -> pd.DataFrame:
    """
    Convert polars DataFrame to pandas, preserving categorical levels from original data.

    Required for patsy/statsmodels which checks categorical levels match between
    training and prediction data.

    Parameters
    ----------
    pred_data : pl.DataFrame
        The prediction data to convert
    original_data : pl.DataFrame
        The original training data with full categorical levels

    Returns
    -------
    pd.DataFrame with categorical columns having correct levels
    """
    pred_pd = pred_data.to_pandas()
    orig_pd = original_data.to_pandas()

    for col in pred_pd.columns:
        if col in orig_pd.columns and hasattr(orig_pd[col], "cat"):
            pred_pd[col] = pd.Categorical(pred_pd[col], categories=orig_pd[col].cat.categories)

    return pred_pd


def make_train(
    data: pl.DataFrame | pd.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
    strat_var: str | list[str] = None,
    test_size: float = 0.2,
    random_state: int = 1234,
):
    """
    If a stratification variable has been provide, will use stratified sampling on one or more variables to create a training variable for your dataset.
    """
    data = check_dataframe(data)

    if strat_var:
        training_var = np.zeros(data.height)
        # StratifiedShuffleSplit needs pandas/numpy
        strat_values = data.select(strat_var).to_numpy().ravel()
        splits = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in splits.split(training_var, strat_values):
            training_var[train_index] = 1
    else:
        training_var = np.random.choice([0, 1], size=data.height, p=[test_size, 1 - test_size])

    return pl.Series(training_var).cast(int)


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
    if v is None:
        return []
    elif isinstance(v, list):
        return v
    elif isinstance(v, str):
        return [v]
    else:
        return list(v)


def convert_binary(data, rvar, lev):
    """Convert response variable to binary (0/1) based on level. Works with polars DataFrame."""
    cb = pl.when(pl.col(rvar) == lev).then(1).when(pl.col(rvar).is_null()).then(None).otherwise(0)
    return data.with_columns(cb.alias(rvar))


def is_binary(series):
    """
    Efficiently check if a series has more than 2 unique values.
    Works with both polars Series and pandas Series.
    """
    if isinstance(series, pl.Series):
        return series.drop_nulls().n_unique() > 2
    else:
        # pandas Series
        return series.dropna().nunique() > 2


def check_binary(data, rvar, lev):
    """Check and convert binary response variable. Works with polars DataFrame."""
    unique_levels = data[rvar].unique().to_list()
    data = convert_binary(data, rvar, lev)
    rvar_sum = data[rvar].sum()
    if rvar_sum == 0 or rvar_sum == data.height:
        raise ValueError(
            f"All converted response values are {1 if rvar_sum == data.height else 0}. "
            f"Available levels in {rvar}: {unique_levels} and '{lev}' was selected."
        )
    else:
        return data


def get_dummies(
    df: pd.DataFrame | pl.DataFrame,
    drop_first: bool = True,
    drop_nonvarying: bool = True,
    categories: dict[str, list] | None = None,
) -> pl.DataFrame:
    """
    Create dummy variables matching pandas get_dummies behavior.

    Categories are sorted alphabetically and the first is dropped (when drop_first=True).
    Numeric columns are placed first, dummy columns appended at end.

    Parameters
    ----------
    df : DataFrame
        Input data (pandas or polars)
    drop_first : bool
        Drop first category alphabetically (default True)
    drop_nonvarying : bool
        If False, preserve all categories even if not present in data.
        Requires categories dict to know which categories to preserve.
    categories : dict[str, list] | None
        Category lists per column from training. Used during prediction.

    Returns
    -------
    pl.DataFrame with dummy columns appended after non-categorical columns
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    cat_cols = [c for c in df.columns if df[c].dtype == pl.Utf8 or df[c].dtype == pl.Categorical]

    if not cat_cols:
        return df

    # Get non-categorical columns (keep at front, like pandas)
    non_cat_cols = [c for c in df.columns if c not in cat_cols]
    result = df.select(non_cat_cols) if non_cat_cols else pl.DataFrame()

    # When categories provided (prediction mode), create dummies directly
    if categories:
        for col in cat_cols:
            if col in categories:
                # Create dummy columns for each category in stored list
                for cat in categories[col]:
                    col_name = f"{col}_{cat}"
                    dummy_col = (df[col].cast(pl.Utf8) == cat).cast(pl.UInt8).alias(col_name)
                    result = result.with_columns(dummy_col)
            else:
                # Column not in categories - create dummies with alphabetical sorting
                cats_sorted = sorted(df[col].cast(pl.Utf8).unique().drop_nulls().to_list())
                cats_to_use = cats_sorted[1:] if drop_first else cats_sorted
                for cat in cats_to_use:
                    col_name = f"{col}_{cat}"
                    dummy_col = (df[col].cast(pl.Utf8) == cat).cast(pl.UInt8).alias(col_name)
                    result = result.with_columns(dummy_col)
        return result

    # Training mode: create dummies with alphabetical sorting (like pandas)
    for col in cat_cols:
        cats_sorted = sorted(df[col].cast(pl.Utf8).unique().drop_nulls().to_list())
        cats_to_use = cats_sorted[1:] if drop_first else cats_sorted
        for cat in cats_to_use:
            col_name = f"{col}_{cat}"
            dummy_col = (df[col].cast(pl.Utf8) == cat).cast(pl.UInt8).alias(col_name)
            result = result.with_columns(dummy_col)

    return result


def conditional_get_dummies(
    df: pd.DataFrame | pl.DataFrame,
    drop_nonvarying: bool = True,
    categories: dict[str, list] | None = None,
) -> pl.DataFrame:
    """
    Create dummy variables conditionally (drop_first only for 3+ categories).

    Parameters
    ----------
    df : DataFrame
        Input data (pandas or polars)
    drop_nonvarying : bool
        If False, preserve all categories even if not present in data.
        Requires categories dict to know which categories to preserve.
    categories : dict[str, list] | None
        Full category lists per column. Used when drop_nonvarying=False.

    Returns
    -------
    pl.DataFrame with dummy columns replacing categorical columns
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    cat_cols = [c for c in df.columns if df[c].dtype == pl.Utf8 or df[c].dtype == pl.Categorical]

    if not cat_cols:
        return df

    # Split into binary (keep all) and non-binary (drop first)
    binary_cols = []
    nonbinary_cols = []
    for col in cat_cols:
        n_cats = df[col].n_unique()
        if n_cats <= 2:
            binary_cols.append(col)
        else:
            nonbinary_cols.append(col)

    # Get non-categorical columns
    non_cat_cols = [c for c in df.columns if c not in cat_cols]
    result = df.select(non_cat_cols) if non_cat_cols else pl.DataFrame()

    # Process binary columns (keep all categories)
    if binary_cols:
        binary_result = get_dummies(
            df.select(binary_cols),
            drop_first=False,
            drop_nonvarying=drop_nonvarying,
            categories={k: v for k, v in (categories or {}).items() if k in binary_cols},
        )
        result = (
            pl.concat([result, binary_result], how="horizontal")
            if result.width > 0
            else binary_result
        )

    # Process non-binary columns (drop first)
    if nonbinary_cols:
        nonbinary_result = get_dummies(
            df.select(nonbinary_cols),
            drop_first=True,
            drop_nonvarying=drop_nonvarying,
            categories={k: v for k, v in (categories or {}).items() if k in nonbinary_cols},
        )
        result = (
            pl.concat([result, nonbinary_result], how="horizontal")
            if result.width > 0
            else nonbinary_result
        )

    return result


# def sig_stars(pval):
#     pval = np.nan_to_num(pval, nan=1.0)
#     cutpoints = np.array([0.001, 0.01, 0.05, 0.1, np.inf])
#     symbols = np.array(["***", "**", "*", ".", " "])
#     return [symbols[p < cutpoints][0] for p in pval]


def sig_stars(pval) -> pl.Series:
    """
    pval: list of floats/None or a pl.Series
    returns: pl.Series of significance symbols (strings)
    """
    # Build a DataFrame so we can use the expression API
    df = pl.DataFrame({"pval": pval}).fill_nan(1)

    # NaN -> 1.0 (like np.nan_to_num(..., nan=1.0)), null stays null
    # Then map to significance stars using when/then/otherwise
    return df.with_columns(
        pl.when(pl.col("pval") < 0.001)
        .then(pl.lit("***"))
        .when(pl.col("pval") < 0.01)
        .then(pl.lit("**"))
        .when(pl.col("pval") < 0.05)
        .then(pl.lit("*"))
        .when(pl.col("pval") < 0.1)
        .then(pl.lit("."))
        .otherwise(pl.lit(" "))
        .alias("sig")
    )["sig"]


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
    df["OR%"] = [f"{round(o, max(dec - 2, 0))}%" for o in df["OR%"]]
    df = df.reset_index()

    if importance:
        df["dummy"] = df["index"].str.contains("[T", regex=False)
        df["importance"] = pd.DataFrame().assign(OR=df["OR"], ORinv=1 / df["OR"]).max(axis=1)

    if isinstance(data, pd.DataFrame):
        # using a fake response variable variable
        data = data.assign(__rvar__=1).copy()
        form = "__rvar__ ~ " + fitted.model.formula.split("~", 1)[1]
        exog = pd.DataFrame(smf.logit(formula=form, data=data).exog)
        weights = fitted._freq_weights
        if sum(weights) > len(weights):

            def wmean(x):
                return weighted_mean(pd.DataFrame(x), weights)[0]

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

    if excl is not None and excl != []:
        excl = ifelse(isinstance(excl, str), [excl], excl)
        rx = "(" + "|".join([f"^{v}$|^{v}\\[" for v in excl]) + ")"
        excl = df["index"].str.match(rf"{rx}")
        if intercept:
            excl[0] = False
        df = df[~excl]

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]

    # Add columns for error bar bounds
    df["xmin"] = df[f"{low}%"]
    df["xmax"] = df[f"{high}%"]

    # Create ordered categorical for y-axis (maintain row order)
    df["index"] = pd.Categorical(df["index"], categories=df["index"].tolist(), ordered=True)

    return (
        ggplot(df, aes(x="OR", y="index"))
        + geom_vline(xintercept=1, linetype="dashdot", color="gray")
        + geom_errorbarh(aes(xmin="xmin", xmax="xmax"), height=0.2)
        + geom_point()
        + scale_x_log10()
        + labs(x="Odds-ratio", y="")
        + theme_bw()
    )


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

    is_categorical = {item.split("[T.")[0]: ("[T." in item) for item in model.exog_names}
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
        vif = [variance_inflation_factor(model.exog, i) for i in range(model.exog.shape[1])]
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
        raise ValueError("Confidence level (conf) must be a numeric value between 0 and 1")

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
                f"{low * 100:.2f}%": expit(Xb - me),
                f"{high * 100:.2f}%": expit(Xb + me),
            }
        )
    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        return pd.DataFrame(
            {
                "prediction": prediction,
                f"{low * 100:.2f}%": Xb - me,
                f"{high * 100:.2f}%": Xb + me,
            }
        )


def nobs_dropped(obj):
    if hasattr(obj, "nobs_dropped") and obj.nobs_dropped > 0:
        return f" ({obj.nobs_dropped:,.0f} obs. dropped)"
    else:
        return ""


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
        weighted_nobs = fitted.df_resid + fitted.df_model + int(hasattr(fitted.params, "Intercept"))
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
Chi-squared: {mfit.chisq.values[0].round(dec)}, df({mfit.chisq_df.values[0]}), p.value {np.where(mfit.chisq_pval.values[0] < 0.001, "< 0.001", mfit.chisq_pval.values[0].round(dec))}
Nr obs: {mfit.nobs.values[0]:,.0f}{nobs_dropped(fitted)}"""
        elif model_type == "regression":
            output = f"""R-squared: {mfit.rsq.values[0].round(dec)}, Adjusted R-squared: {mfit.rsq_adj.values[0].round(dec)}
F-statistic: {mfit.fvalue[0].round(dec)} df({mfit.ftest_df_model.values[0]:.0f}, {mfit.ftest_df_resid.values[0]:.0f}), p.value {np.where(mfit.ftest_pval.values[0] < 0.001, "< 0.001", mfit.ftest_pval.values[0].round(dec))}
Nr obs: {mfit.nobs.values[0]:,.0f}{nobs_dropped(fitted)}"""
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

    # Add columns for error bar bounds
    df["xmin"] = df[f"{low}%"]
    df["xmax"] = df[f"{high}%"]

    # Create ordered categorical for y-axis (maintain row order)
    df["index"] = pd.Categorical(df["index"], categories=df["index"].tolist(), ordered=True)

    return (
        ggplot(df, aes(x="coefficient", y="index"))
        + geom_vline(xintercept=0, linetype="dashdot", color="gray")
        + geom_errorbarh(aes(xmin="xmin", xmax="xmax"), height=0.2)
        + geom_point()
        + labs(x="Coefficient", y="")
        + theme_bw()
    )


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
        df["p.values"] = ifelse(fitted.pvalues < 0.001, "< .001", fitted.pvalues.round(dec))

    df["  "] = sig_stars(fitted.pvalues)
    if intercept is False:
        df = df[df.index != "Intercept"]

    return df


def evalreg(df, rvar: str, pred: str, dec: int = 3):
    """
    Evaluate regression models. Calculates R-squared, MSE, and MAE

    Parameters
    ----------
    df : DataFrame (polars or pandas) or a dictionary of DataFrames with keys to show results for
        multiple model predictions and datasets (training and test)
    rvar : str
        Name of the response variable column in df
    pred : str
        Name of the column, of list of column names, in df with model predictions
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with evaluation metrics
    """

    dct = ifelse(isinstance(df, dict), df, {"All": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)

    def calculate_metrics(key, dfm, pm):
        dfm = check_dataframe(dfm)
        n = dfm.height
        y_true = dfm[rvar].to_numpy()
        y_pred = dfm[pm].to_numpy()
        return pl.DataFrame(
            {
                "Type": [key],
                "predictor": [pm],
                "n": [n],
                "r2": [metrics.r2_score(y_true, y_pred)],
                "mse": [metrics.mean_squared_error(y_true, y_pred)],
                "mae": [metrics.mean_absolute_error(y_true, y_pred)],
            }
        )

    result = pl.concat([calculate_metrics(key, val, p) for key, val in dct.items() for p in pred])
    # Round numeric columns
    numeric_cols = [c for c in result.columns if result[c].dtype == pl.Float64]
    return result.with_columns([pl.col(c).round(dec) for c in numeric_cols])


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
    if hasattr(fitted.model, "endog"):
        endog = fitted.model.endog
    else:
        endog = fitted.model["endog"]

    data = pl.DataFrame(
        {
            "fitted": fitted.fittedvalues,
            "actual": endog,
            "resid": fitted.resid,
            "std_resid": fitted.resid / fitted.resid.std(),
            "order": np.arange(endog.shape[0]),
        }
    )

    if (nobs != -1 and nobs is not None) and (nobs < data.height):
        data = data.sample(nobs, seed=1234)

    # Plot 1: Actual vs Fitted values
    p1 = (
        ggplot(data, aes(x="fitted", y="actual"))
        + geom_point(alpha=0.5)
        + geom_smooth(method="lm", color="blue")
        + labs(x="Fitted values", y="Actual values")
        + ggtitle("Actual vs Fitted values")
        + theme_bw()
    )

    # Plot 2: Residuals vs Fitted values
    p2 = (
        ggplot(data, aes(x="fitted", y="resid"))
        + geom_point(alpha=0.5)
        + geom_smooth(method="lm", color="blue")
        + geom_hline(yintercept=0, linetype="dashed", color="gray")
        + labs(x="Fitted values", y="Residuals")
        + ggtitle("Residuals vs Fitted values")
        + theme_bw()
    )

    # Plot 3: Residuals vs Row order
    p3 = (
        ggplot(data, aes(x="order", y="resid"))
        + geom_line()
        + geom_hline(yintercept=0, linetype="dashed", color="gray")
        + labs(x="Row order", y="Residuals")
        + ggtitle("Residuals vs Row order")
        + theme_bw()
    )

    # Plot 4: Q-Q plot
    p4 = (
        ggplot(data, aes(sample="std_resid"))
        + stat_qq()
        + stat_qq_line()
        + labs(x="Theoretical quantiles", y="Sample quantiles")
        + ggtitle("Normal Q-Q plot")
        + theme_bw()
    )

    # Plot 5: Histogram of residuals
    p5 = (
        ggplot(data, aes(x="resid"))
        + geom_histogram(fill="slateblue", color="white", bins=30)
        + labs(x="Residuals", y="Count")
        + ggtitle("Histogram of residuals")
        + theme_bw()
    )

    # Plot 6: Residuals vs Normal density
    p6 = (
        ggplot(data, aes(x="std_resid"))
        + geom_density(fill="green", alpha=0.5)
        + stat_function(fun=stats.norm.pdf, color="black", size=1)
        + labs(x="Standardized residuals", y="Density")
        + ggtitle("Residuals vs Normal density")
        + theme_bw()
    )

    # Compose into 3x2 grid and set figure size
    dashboard = (p1 | p2) / (p3 | p4) / (p5 | p6) + theme(figure_size=(10, 12))
    return dashboard


def sim_prediction(
    data: pd.DataFrame | pl.DataFrame,
    vary: list = [],
    nnv: int = 5,
    minq: float = 0,
    maxq: float = 1,
) -> pl.DataFrame:
    """
    Simulate data for prediction

    Parameters
    ----------
    data : DataFrame (polars or pandas)
    vary : List of column names or Dictionary with keys and values to use
    nnv : int
        Number of values to use to simulate the effect of a numeric variable
    minq : float
        Quantile to use for the minimum value of numeric variables
    maxq : float
        Quantile to use for the maximum value of numeric variables

    Returns:
    ----------
    pl.DataFrame with values to use for prediction
    """
    # Convert to polars
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    def fix_value(col_name):
        col = data[col_name]
        if col.dtype.is_numeric():
            return col.mean()
        else:
            # Get most common value
            return col.value_counts().sort("count", descending=True)[0, col_name]

    dct = {c: [fix_value(c)] for c in data.columns}
    schema = {c: data[c].dtype for c in data.columns}

    if isinstance(vary, dict):
        # user provided values and ranges
        for key, val in vary.items():
            dct[key] = val
    else:
        # auto derived values and ranges
        vary = ifelse(isinstance(vary, str), [vary], vary)
        for v in vary:
            col = data[v]
            if col.dtype.is_numeric():
                nu = col.n_unique()
                if nu > 2:
                    min_val = col.quantile(minq)
                    max_val = col.quantile(maxq)
                    dct[v] = np.linspace(min_val, max_val, min([nu, nnv])).tolist()
                else:
                    dct[v] = [col.min(), col.max()]
            else:
                dct[v] = col.unique().to_list()

    return expand_grid(dct, schema)


def scatter_plot(fitted, df, nobs: int = 1000, figsize: tuple = None, resid=False) -> None:
    """
    Scatter plot of explanatory and response variables from a fitted regression

    Parameters
    ----------
    fitted : A fitted linear regression model
    df : DataFrame (polars or pandas)
        Data frame with explanatory and response variables
    nobs : int
        Number of observations to use for the scatter plots. The default
        value is 1,000. To use all observations in the plots, use nobs=-1
    figsize : tuple
        A tuple that determines the figure size (ignored, kept for compatibility)
    resid : bool
        If True, use residuals as the response variable
    """
    # Convert to polars
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    exog_names = extract_evars(fitted.model, df.columns)

    if resid:
        endog_name = "residuals"
        if df.height != len(fitted.resid):
            raise ValueError(
                "The number of observations in the fitted model and the data frame must be the same"
            )
        df = df.select(exog_names).with_columns(pl.Series(endog_name, fitted.resid))
    else:
        endog_name = extract_rvar(fitted.model, df.columns)
        df = df.select([endog_name] + exog_names)

    if nobs < df.height and nobs != np.inf and nobs != -1:
        df = df.sample(nobs, seed=1234)

    plots = []
    for exog_name in exog_names:
        col = df[exog_name]
        if col.dtype.is_numeric():
            # Scatter plot for numeric variables
            p = (
                ggplot(df, aes(x=exog_name, y=endog_name))
                + geom_point(alpha=0.5)
                + labs(x=exog_name, y=endog_name)
                + theme_bw()
            )
        else:
            # Jitter plot for categorical variables with mean lines
            means_df = df.group_by(exog_name).agg(pl.col(endog_name).mean().alias("mean"))

            p = (
                ggplot(df, aes(x=exog_name, y=endog_name))
                + geom_jitter(alpha=0.5, width=0.2)
                + geom_point(
                    data=means_df,
                    mapping=aes(x=exog_name, y="mean"),
                    color="blue",
                    size=3,
                )
                + geom_segment(
                    data=means_df,
                    mapping=aes(x=exog_name, xend=exog_name, y="mean", yend="mean"),
                    color="blue",
                    size=2,
                )
                + labs(x=exog_name, y=endog_name)
                + theme_bw()
            )
        plots.append(p)

    # Compose plots into a grid (2 columns)
    if len(plots) == 1:
        result = plots[0]
    else:
        # Build rows of 2 plots each
        rows = []
        for i in range(0, len(plots), 2):
            if i + 1 < len(plots):
                rows.append(plots[i] | plots[i + 1])
            else:
                rows.append(plots[i])

        # Stack rows vertically
        result = rows[0]
        for row in rows[1:]:
            result = result / row

        # Set figure size for composition
        nr_rows = len(rows)
        result = result + theme(figure_size=(10, 3 * nr_rows))

    return result


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

    return scatter_plot(fitted, df, nobs=nobs, figsize=figsize, resid=True)


def cross_validation(
    mlobj,
    name,
    param_grid,
    scoring=None,
    prn=True,
    directory="cv-objects",
    n_splits=5,
    n_jobs=2,
    verbose=1,
    random_state=1234,
):
    """
    Use cross-validation to find the best hyper parameters for a machine learning model. This function can be used with the pyrsm package to automatically save the cross-validation object to a file and load it if it already exists.

    Parameters
    ----------
    mlobj: An object created by the pyrsm package with a fitted machine learning model
    name: str
        Name of the model to use in the file name for the cross-validation object. Most likely the same as the name of the model object
    param_grid: dict
        Dictionary with parameters to use for cross-validation
    scoring: dict
        Dictionary with scoring metrics to use for cross-validation. Defaults to {"r2": "r2", "rmse": "neg_root_mean_squared_error"} for regression and {"AUC": "roc_auc", "accuracy": "accuracy"} for classification
    prn : bool
        If True print output from cross-validation
    directory : str
        Directory to save the cross-validation object. Defaults to "cv-objects"
    n_splits : int
        Number of splits to use in cross validation. Defaults to 5.
    n_jobs : int
        Number of just to start to run the cross validation. Defaults to 2.
    random_state : int:
        Random seed to use. Defaults to 1234.

    Returns:
        cv: An sklearn GridSearchCV object
    """

    if not hasattr(mlobj, "mod_type"):
        raise ValueError("mlobj must be a pyrsm object")
    if not hasattr(mlobj, "fitted"):
        raise ValueError("mlobj must be a pyrsm object with a fitted model")
    if not hasattr(mlobj, "data_onehot"):
        raise ValueError("mlobj must be a pyrsm object with data_onehot")

    if mlobj.mod_type == "classification":
        kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if scoring is None:
            scoring = {"AUC": "roc_auc", "accuracy": "accuracy"}
    elif mlobj.mod_type == "regression":
        kfolds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if scoring is None:
            scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
    else:
        raise ValueError("mlobj must be a pyrsm object with a fitted model")

    refit = list(scoring.keys())[0]

    cv_file = f"{directory}/{name}-cross-validation-object.pkl"
    if os.path.exists(cv_file):
        cv = load_state(cv_file)["cv"]
    else:
        rvar = mlobj.rvar
        cv = GridSearchCV(
            mlobj.fitted,
            param_grid=param_grid,
            scoring=scoring,
            cv=kfolds,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
        ).fit(mlobj.data_onehot, mlobj.data[rvar])
        if not os.path.exists(directory):
            os.mkdir(directory)
        save_state({"cv": cv}, cv_file)

    if prn:
        print("The best parameters are:", cv.best_params_)
        print("The best model fit score is:", cv.best_score_)
        print(
            "The GirdSearchCV model fit estimates:\n",
            pd.DataFrame(cv.cv_results_).sort_values(f"rank_test_{refit}").head(),
        )

    return cv


# %%
