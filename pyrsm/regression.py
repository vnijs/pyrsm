from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from scipy import stats
import statsmodels as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResults as rrs
from math import ceil
from .logit import sig_stars
from .utils import setdiff, format_nr, expand_grid, ifelse, group_categorical


def coef_plot(
    fitted,
    alpha: float = 0.05,
    intercept: bool = False,
    incl: str = None,
    excl: str = None,
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
        incl = ifelse(isinstance(incl, list), incl, [incl])
        rx = "(" + "|".join([f"^\b{v}|^{v}\\[" for v in incl]) + ")"
        incl = df["index"].str.match(rf"{rx}")
        if intercept:
            incl[0] = True
        df = df[incl]

    if excl is not None:
        excl = ifelse(isinstance(excl, list), excl, [excl])
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


def coef_ci(fitted, alpha: float = 0.05, intercept: bool = False, dec: int = 3):
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
    smf.qqplot(data.resid, line="s", ax=axes[1, 1])
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


def sim_prediction(
    df: pd.DataFrame,
    vary: list = [],
    nnv: int = 5,
    minq: float = 0,
    maxq: float = 1,
) -> pd.DataFrame:
    """
    Simulate data for prediction

    Parameters
    ----------
    df : Pandas DataFrame
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

    dct = {c: [fix_value(df[c])] for c in df.columns}
    dt = df.dtypes
    if type(vary) is dict:
        # user provided values and ranges
        for key, val in vary.items():
            dct[key] = val
    else:
        # auto derived values and ranges
        vary = ifelse(type(vary) is list, vary, [vary])
        for v in vary:
            if pd.api.types.is_numeric_dtype(df[v].dtype):
                nu = df[v].nunique()
                if nu > 2:
                    dct[v] = np.linspace(
                        np.quantile(df[v], minq),
                        np.quantile(df[v], maxq),
                        min([nu, nnv]),
                    )
                else:
                    dct[v] = [df[v].min(), df[v].max()]
            else:
                dct[v] = df[v].unique()

    return expand_grid(dct, dt)


def regress(
    dataset: pd.DataFrame,
    rvar: str = None,
    evars: list[str] = None,
    form: str = "",
    ssq: bool = False,
) -> rrs:
    """
    Estimate linear regression model

    Parameters
    ----------
    dataset: pandas DataFrame; dataset
    evars: List of strings; contains the names of the columns of data to be used as explanatory variables
    rvar: String; name of the column to be used as the response variable
    form: String; formula for the regression equation to use if evars and rvar are not provided

    Returns
    -------
    res: Object with fitted values and residuals
    """
    if form != "":
        model = smf.ols(form, data=dataset)
        evars = setdiff(model.exog_names, "const")
        rvar = model.endog_names
    else:
        form = f"{rvar} ~ {' + '.join(evars)}"
        model = smf.ols(form, data=dataset)

    res = model.fit()

    data_name = ""
    if hasattr(dataset, "description"):
        data_name = dataset.description.split("\n")[0].split()[1].lower()

    print("Data: ", data_name)
    print("Response variable    :", rvar)
    print("Explanatory variables:", ", ".join(evars))
    print(f"Null hyp.: the effect of x on {rvar} is zero")
    print(f"Alt. hyp.: the effect of x on {rvar} is not zero")
    summary = res.summary()
    summary.tables.pop()  # gets rid of the last table in res.summary(), we don't need it
    print("\n", summary)

    if ssq:
        print("Sum of squares")
        index = ["Regression", "Error", "Total"]
        sum_of_squares = [res.ess, res.ssr, res.centered_tss]
        sum_of_squares_series = pd.Series(
            data=format_nr(sum_of_squares, dec=0), index=index
        )
        print(f"\n{sum_of_squares_series.to_string()}")

    return res


def scatter_plot(fitted, nobs: int = 1000, figsize: tuple = None) -> None:
    # TODO: this has some bug when dealing with the plotting of categorical variables
    """
    Scatter plot of explanatory and response variables from a fitted regression

    Parameters
    ----------
    nobs : int
        Number of observations to use for the scatter plots. The default
        value is 1,000. To use all observations in the plots, use nobs=-1
    figsize : tuple
        A tuple that determines the figure size. If None, size is
        determined based on the number of variables in the model
    """

    endog = fitted.model.endog
    exogs = fitted.model.exog

    exog_names = fitted.model.exog_names
    endog_name = fitted.model.endog_names

    print(f"exog_names: {exog_names}")
    print()

    df = pd.DataFrame(exogs, columns=exog_names)
    df, was_grouped = group_categorical(df)
    df.drop("Intercept", axis=1, inplace=True)
    exog_names = df.columns.to_list()
    print(f"exog_names: {exog_names}")
    if was_grouped:
        exogs = df[exog_names].to_numpy()

    num_exog = len(exog_names)
    num_rows = ceil(num_exog / 2)

    if figsize is None:
        figsize = (num_rows * 5, max(5, min(num_exog, 2) * 5))

    fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    idx = 0

    if nobs < fitted.model.endog.shape[0] and nobs != np.Inf and nobs != -1:
        df[endog_name] = endog

        df = df.copy().sample(nobs)

        endog = df[endog_name].to_numpy()
        exogs = df[exog_names].to_numpy()

    while idx < num_exog:
        row = idx // 2
        col = idx % 2
        exog_name = exog_names[idx]
        exog = [row[idx] for row in exogs]

        if num_rows > 1:
            axes[row][col].set_xlabel(exog_name)
            axes[row][col].set_ylabel(endog_name)
            axes[row][col].scatter(exog, endog)
        else:
            axes[col].set_xlabel(exog_name)
            axes[col].set_ylabel(endog_name)
            axes[col].scatter(exog, endog)
        idx += 1

    if (df.shape[1] - 1) % 2 != 0:
        if num_rows > 1:
            fig.delaxes(axes[row][1])  # remove last empty plot
        else:
            fig.delaxes(axes[1])

    plt.show()


def residual_vs_explanatory_plot(
    fitted: rrs,
    nobs: int = 1000,
    figsize: Tuple = None,
) -> None:
    """
    Plot of variables vs residuals

    Parameters
    ----------
    nobs : int
        Number of observations to use for the scatter plots. The default
        value is 1,000. To use all observations in the plots, use nobs=-1
    figsize : tuple
        A tuple that determines the figure size. If None, size is
        determined based on the number of variables in the model
    """

    exogs = fitted.model.exog

    exog_names = fitted.model.exog_names

    df = pd.DataFrame(exogs, columns=exog_names)
    df, _ = group_categorical(df)
    df.drop("Intercept", axis=1, inplace=True)
    true_exog_names = df.columns.to_list()

    num_exog = len(true_exog_names)
    num_rows = ceil(num_exog / 2)

    if figsize is None:
        figsize = (num_rows * 5, max(5, min(num_exog, 2) * 5))

    fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    idx = 0

    residuals = fitted.resid

    data = pd.DataFrame(exogs, columns=exog_names)
    data, _ = group_categorical(data)

    data["residuals"] = residuals

    if nobs < fitted.model.endog.shape[0] and nobs != np.Inf and nobs != -1:
        data = data.copy().sample(nobs)

    categorical_dtypes = ["object", "category", "bool"]

    while idx < num_exog:
        row = idx // 2
        col = idx % 2
        exog_name = true_exog_names[idx]

        if num_rows > 1:
            if data[exog_name].dtype.name in categorical_dtypes:
                sns.stripplot(
                    x=exog_name,
                    y="residuals",
                    data=data,
                    ax=axes[row][col],
                    c="black",
                ).set(xlabel=exog_name, ylabel="Residuals")
            else:
                sns.regplot(
                    x=exog_name,
                    y="residuals",
                    data=data,
                    ax=axes[row][col],
                    scatter_kws={"color": "black"},
                ).set(xlabel=exog_name, ylabel="Residuals")
        else:
            if data[exog_name].dtype.name in categorical_dtypes:
                sns.stripplot(
                    x=exog_name,
                    y="residuals",
                    data=data,
                    ax=axes[col],
                    c="black",
                ).set(xlabel=exog_name, ylabel="Residuals")
            else:
                sns.regplot(
                    x=exog_name,
                    y="residuals",
                    data=data,
                    ax=axes[col],
                    scatter_kws={"color": "black"},
                ).set(xlabel=exog_name, ylabel="Residuals")
        idx += 1

    if df.shape[1] % 2 != 0:
        if num_rows > 1:
            fig.delaxes(axes[row][1])  # remove last empty plot
        else:
            fig.delaxes(axes[1])  # remove last empty plot

    plt.show()
