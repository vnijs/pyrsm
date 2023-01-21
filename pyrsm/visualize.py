import math
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels as sm
from sklearn.inspection import permutation_importance
from .utils import ifelse
from .regression import sim_prediction
from .perf import auc


def distr_plot(df: pd.DataFrame, cols: list = None, nint: int = 25, **kwargs):
    """
    Plot histograms for numeric variables and frequency plots for categorical.
    variables. Columns of type integer with less than 25 unique values will be
    treated as categorical. To change this behavior, increase or decrease the
    value of the 'nint' argument

    Parameters
    ----------
    df : Pandas dataframe
    cols: A list of column names to generate distribution plots for. If None, all
        variables will be plotted
    nint: int
        The number of unique values in a series of type integer below which the
        series will be treated as a categorical variable
    **kwargs : Named arguments to be passed to the pandas plotting methods
    """
    if cols is None:
        all_cols = df.columns
        cols = []
        for i, c in enumerate(all_cols):
            if (
                not pd.api.types.is_numeric_dtype(df[c].dtype)
                and not pd.api.types.is_categorical_dtype(df[c].dtype)
            ) or pd.api.types.is_object_dtype(df[c].dtype):
                print(f"No plot will be created for {c} (type {df[c].dtype})")
            else:
                cols.append(c)

    df = df.loc[:, cols].copy()

    fig, axes = plt.subplots(
        max(math.ceil(df.shape[1] / 2), 2), 2, figsize=(10, 2 * max(df.shape[1], 4))
    )
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    row = 0
    for i, c in enumerate(df.columns):
        s = df[c]
        j = ifelse(i % 2 == 0, 0, 1)
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique() < nint:
            s.sort_values().value_counts(sort=False).plot.bar(
                ax=axes[row, j], title=c, rot=0, color="slateblue", **kwargs
            )
        elif pd.api.types.is_numeric_dtype(s.dtype):
            s.plot.hist(ax=axes[row, j], title=c, rot=0, color="slateblue", **kwargs)
        elif pd.api.types.is_categorical_dtype(s.dtype):
            s.value_counts(sort=False).plot.bar(
                ax=axes[row, j], title=c, rot=0, color="slateblue", **kwargs
            )
        else:
            print(f"No plot will be created for {c} (type {s.dtype})")

        if j == 1:
            row += 1

    # needed because the axes object must always be 2-dimensional
    # or else the indexing used in this function will fail for small
    # DataFrames
    if df.shape[1] < 3:
        axes[-1, -1].remove()
        axes[-1, 0].remove()
        if df.shape[1] == 1:
            axes[0, -1].remove()
    elif df.shape[1] % 2 == 1:
        axes[-1, -1].remove()

    plt.show()


def scatter(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    _, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.scatter(df[col1], df[col2])

    plt.show()


def extract_evars(model, cn):
    """
    Extract a list of the names of the explanatory variables in a statsmodels model
    """
    pattern = r"\b\w+\b"
    evars = re.findall(pattern, model.formula)[1:]
    evars = [v for v in evars if v in cn]
    return [v for i, v in enumerate(evars) if v not in evars[:i]]


def pred_plot_sm(
    fitted,
    df,
    incl=None,
    excl=[],
    incl_int=[],
    fix=True,
    hline=True,
    nnv=20,
    minq=0.025,
    maxq=0.975,
):
    """
    Generate prediction plots for statsmodels regression models (OLS and Logistic).
    A faster alternative to PDP plots.

    Parameters
    ----------
    fitted : A fitted (logistic) regression model
    dataset: pandas DataFrame; dataset
    incl: List of strings; contains the names of the columns of data to use for prediction
          By default it will extract the names of all explanatory variables used in estimation
          Use [] to ensure no single-variable plots are created
    excl: List of strings; contains names of columns to exclude. useful when the list of
          variable names is automatically
    incl_int: List of strings; contains the names of the columns of data to be interacted for
          prediction plotting (e.g., ["x1:x2", "x2:x3"] would generate interaction plots for
          x1 and x2 and x2 and x3
    fix : Logical or tuple
        Set the desired limited on yhat or have it calculated automatically.
        Set to FALSE to have y-axis limits set by ggplot2 for each plot
    hline : Logical or float
        Add a horizontal line at the average of the target variable. When set to False
        no line is added. When set to a specific number, the horizontal line will be
        added at that value
    nnv: Integer: The number of values to simulate for numeric variables used in prediction
    minq : float
        Quantile to use for the minimum value for simulation of numeric variables
    maxq : float
        Quantile to use for the maximum value for simulation of numeric variables

    Examples
    -------
    pred_plot(lr, df, excl="monetary")
    pred_plot(lr, df, incl = [], incl_int = ["frequency:monetary"])
    """
    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        # legacy for when only an un-fitted model was accepted
        model = fitted

    min_max = [np.Inf, -np.Inf]

    def calc_ylim(lab, lst, min_max):
        if isinstance(fix, bool) and fix == True:
            vals = lst[lab]
            return (min(min_max[0], min(vals)), max(min_max[1], max(vals)))
        elif not isinstance(fix, bool) and len(fix) == 2:
            return fix
        else:
            return False

    rvar = model.endog_names
    if isinstance(hline, bool) and hline == True:
        hline = df[rvar].mean()

    if not isinstance(hline, float) and not isinstance(hline, int):
        hline = False

    if incl is None:
        incl = extract_evars(model, df.columns)
    else:
        incl = ifelse(isinstance(incl, list), incl, [incl])

    excl = ifelse(isinstance(excl, list), excl, [excl])
    incl_int = ifelse(isinstance(incl_int, list), incl_int, [incl_int])

    if len(excl) > 0:
        incl = [i for i in incl if i not in excl]

    nr_plots = len(incl) + len(incl_int)
    fig, ax = plt.subplots(
        max(math.ceil(nr_plots / 2), 2), 2, figsize=(10, 2 * max(nr_plots, 4))
    )
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    pred_dict = {}
    for v in incl:
        iplot = sim_prediction(df, vary=v, nnv=nnv, minq=minq, maxq=maxq)
        iplot["prediction"] = fitted.predict(iplot)
        min_max = calc_ylim("prediction", iplot, min_max)
        pred_dict[v] = iplot

    for v in incl_int:
        vl = v.split(":")
        is_num = [pd.api.types.is_numeric_dtype(df[c].dtype) for c in vl]
        iplot = sim_prediction(df, vary=vl, nnv=nnv, minq=minq, maxq=maxq)
        iplot["prediction"] = fitted.predict(iplot)
        if sum(is_num) < 2:
            min_max = calc_ylim("prediction", iplot, min_max)
        pred_dict[v] = iplot

    row = col = 0
    for i, v in enumerate(incl):
        col = ifelse(i % 2 == 0, 0, 1)
        if pd.api.types.is_numeric_dtype(df[v].dtype):
            fig = sns.lineplot(x=v, y="prediction", data=pred_dict[v], ax=ax[row, col])
        else:
            fig = sns.lineplot(
                x=v, y="prediction", marker="o", data=pred_dict[v], ax=ax[row, col]
            )
        if isinstance(min_max, tuple) and len(min_max) == 2:
            ax[row, col].set(ylim=tuple(min_max))
        if hline != False:
            ax[row, col].axhline(y=hline, linestyle="--")

        if col == 1:
            row += 1

    start_col = ifelse(col == 1, 0, 1)
    for j, v in enumerate(incl_int):
        col = ifelse(j % 2 == start_col, 0, 1)
        vl = v.split(":")
        is_num = [pd.api.types.is_numeric_dtype(df[c].dtype) for c in vl]
        if sum(is_num) == 2:
            plot_data = (
                pred_dict[v]
                .pivot(vl[0], vl[1], "prediction")
                .transpose()
                .sort_values(vl[1], ascending=False)
            )
            fig = sns.heatmap(
                plot_data, ax=ax[row, col], xticklabels=False, yticklabels=False
            )
            # ax[i+j].imshow(plot_data) # not bad - investigate more
        elif sum(is_num) == 1:
            if is_num[1]:
                vl.reverse()
            fig = sns.lineplot(
                x=vl[0], y="prediction", hue=vl[1], data=pred_dict[v], ax=ax[row, col]
            )
        else:
            fig = sns.lineplot(
                x=vl[0],
                y="prediction",
                hue=vl[1],
                marker="o",
                data=pred_dict[v],
                ax=ax[row, col],
            )

        if isinstance(min_max, tuple) and len(min_max) == 2 and sum(is_num) < 2:
            ax[row, col].set(ylim=tuple(min_max))
        if sum(is_num) < 2 and hline != False:
            ax[row, col].axhline(y=hline, linestyle="--")

        if col == 1:
            row += 1

    # needed because the axes object must always be 2-dimensional
    # or else the indexing used in this function will fail for small
    # DataFrames
    if nr_plots < 3:
        ax[-1, -1].remove()
        ax[-1, 0].remove()
        if nr_plots == 1:
            ax[0, -1].remove()
    elif nr_plots % 2 == 1:
        ax[-1, -1].remove()


def pred_plot_sk(
    fitted,
    df,
    rvar=None,
    incl=None,
    excl=[],
    incl_int=[],
    fix=True,
    hline=True,
    nnv=20,
    minq=0.025,
    maxq=0.975,
):
    """
    Generate prediction plots for sklearn models. A faster alternative to PDP plots
    that can handle interaction plots with categorical variables

    Parameters
    ----------
    fitted : A fitted sklearn model
    df : Pandas DataFrame with data used for estimation
    rvar : The column name for the response/target variable
    incl : A list of column names to generate prediction plots for. If None, all
        variables will be plotted
    excl : A list of column names to exclude from plotting
    incl_int : A list is ":" separated column names to plots interaction plots for.
        For example incl_int = ["a:b", "b:c"] would generate interaction plots for
        variables a x b and b x c
    fix : Logical or tuple
        Set the desired limited on yhat or have it calculated automatically.
        Set to False to have y-axis limits set by ggplot2 for each plot
    hline : Logical or float
        Add a horizontal line at the average of the target variable. When set to False
        no line is added. When set to a specific number, the horizontal line will be
        added at that value
    nnv: int
        The number of values to use in simulation for numeric variables
    minq : float
        Quantile to use for the minimum value of numeric variables
    maxq : float
        Quantile to use for the maximum value of numeric variables
    """
    # features names used in the sklearn model
    fn = fitted.feature_names_in_

    not_transformed = [c for c in df.columns for f in fn if c == f]
    transformed = list(set([c for c in df.columns for f in fn if c in f and c != f]))

    if incl is None:
        incl = not_transformed + transformed
    else:
        incl = ifelse(type(incl) is list, incl, [incl])

    def dummify(df, trs):
        if len(trs) > 0:
            return pd.concat([pd.get_dummies(df[trs]), df.drop(trs, axis=1)], axis=1)
        else:
            return df

    min_max = [np.Inf, -np.Inf]
    if rvar is not None and isinstance(hline, bool) and hline == True:
        hline = df[rvar].mean()

    if not isinstance(hline, float) and not isinstance(hline, int):
        hline = False

    def calc_ylim(lab, lst, min_max):
        if isinstance(fix, bool) and fix == True:
            vals = lst[lab]
            return (min(min_max[0], min(vals)), max(min_max[1], max(vals)))
        elif not isinstance(fix, bool) and len(fix) == 2:
            return fix
        else:
            return False

    excl = ifelse(isinstance(excl, list), excl, [excl])
    incl_int = ifelse(isinstance(incl_int, list), incl_int, [incl_int])

    if len(excl) > 0:
        incl = [i for i in incl if i not in excl]

    nr_plots = len(incl) + len(incl_int)
    fig, ax = plt.subplots(
        max(math.ceil(nr_plots / 2), 2), 2, figsize=(10, 2 * max(nr_plots, 4))
    )
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    pred_dict = {}
    for v in incl:
        iplot = sim_prediction(
            df[transformed + not_transformed].dropna(),
            vary=v,
            nnv=nnv,
            minq=minq,
            maxq=maxq,
        )
        iplot_dum = dummify(iplot, transformed)[fn]
        iplot["prediction"] = fitted.predict_proba(iplot_dum)[:, 1]
        min_max = calc_ylim("prediction", iplot, min_max)
        pred_dict[v] = iplot

    for v in incl_int:
        vl = v.split(":")
        is_num = [pd.api.types.is_numeric_dtype(df[c].dtype) for c in vl]
        iplot = sim_prediction(
            df[transformed + not_transformed].dropna(),
            vary=vl,
            nnv=nnv,
            minq=minq,
            maxq=maxq,
        )
        iplot_dum = dummify(iplot, transformed)[fn]
        iplot["prediction"] = fitted.predict_proba(iplot_dum)[:, 1]
        if sum(is_num) < 2:
            min_max = calc_ylim("prediction", iplot, min_max)
        pred_dict[v] = iplot

    row = col = 0
    for i, v in enumerate(incl):
        col = ifelse(i % 2 == 0, 0, 1)
        if pd.api.types.is_numeric_dtype(df[v].dtype):
            fig = sns.lineplot(x=v, y="prediction", data=pred_dict[v], ax=ax[row, col])
        else:
            fig = sns.lineplot(
                x=v, y="prediction", marker="o", data=pred_dict[v], ax=ax[row, col]
            )
        # fig = sns.lineplot(x=v, y="prediction", data=pred_dict[v], ax=ax[row, col])
        if isinstance(min_max, tuple) and len(min_max) == 2:
            ax[row, col].set(ylim=tuple(min_max))
        if hline != False:
            ax[row, col].axhline(y=hline, linestyle="--")

        if col == 1:
            row += 1

    start_col = ifelse(col == 1, 0, 1)
    for j, v in enumerate(incl_int):
        col = ifelse(j % 2 == start_col, 0, 1)
        vl = v.split(":")
        is_num = [pd.api.types.is_numeric_dtype(df[c].dtype) for c in vl]
        if sum(is_num) == 2:
            plot_data = (
                pred_dict[v]
                .pivot(vl[0], vl[1], "prediction")
                .transpose()
                .sort_values(vl[1], ascending=False)
            )
            fig = sns.heatmap(
                plot_data, ax=ax[row, col], xticklabels=False, yticklabels=False
            )
            # ax[i+j].imshow(plot_data) # not bad - investigate more
        elif sum(is_num) == 1:
            if is_num[1]:
                vl.reverse()
            fig = sns.lineplot(
                x=vl[0], y="prediction", hue=vl[1], data=pred_dict[v], ax=ax[row, col]
            )
        else:
            fig = sns.lineplot(
                x=vl[0],
                y="prediction",
                hue=vl[1],
                marker="o",
                data=pred_dict[v],
                ax=ax[row, col],
            )

        if isinstance(min_max, tuple) and len(min_max) == 2 and sum(is_num) < 2:
            ax[row, col].set(ylim=tuple(min_max))
        if sum(is_num) < 2 and hline != False:
            ax[row, col].axhline(y=hline, linestyle="--")

        if col == 1:
            row += 1

    # needed because the axes object must always be 2-dimensional
    # or else the indexing used in this function will fail for small
    # DataFrames
    if nr_plots < 3:
        ax[-1, -1].remove()
        ax[-1, 0].remove()
        if nr_plots == 1:
            ax[0, -1].remove()
    elif nr_plots % 2 == 1:
        ax[-1, -1].remove()


def vimp_plot_sm(fitted, df, rep=5):
    """
    Creates permutation importance plots for models estimated using the
    statsmodels library

    Parameters
    ----------
    fitted : A fitted statsmodels objects
    df : Pandas DataFrame with data used for estimation
    rep: int
        The number of times to resample and calculate the permutation importance
    """
    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        return "This function requires a fitted linear or logistic regression"

    rvar = model.endog_names
    evars = extract_evars(model, df.columns)
    df = df[[rvar] + evars].copy().reset_index(drop=True).dropna()

    def imp_calc_reg(base, pred):
        return pd.DataFrame({"y": model.endog, "yhat": pred}).corr().iloc[0, 1] ** 2

    def imp_calc_logit(base, pred):
        return base - auc(model.endog, pred)

    # Calculate the baseline mean squared error
    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        baseline_fit = auc(model.endog, fitted.predict(df[evars]))
        imp_calc = imp_calc_logit  # specifying the function to use
        xlab = "Importance (AUC decrease)"
    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        # baseline_fit = mean_squared_error(model.endog, fitted.predict(df[evars]))
        baseline_fit = (
            pd.DataFrame({"y": model.endog, "yhat": fitted.predict(df[evars])})
            .corr()
            .iloc[0, 1]
            ** 2
        )
        imp_calc = imp_calc_reg  # specifying the function to use
        xlab = "Importance (R-square decrease)"
    else:
        return "Model for this model type not supported"

    # Create a copy of the dataframe
    permuted = df.copy()

    # Initialize a dictionary to store the permutation importance values
    importance_values = {v: 0 for v in evars}

    # Iterate over each feature
    for i in range(rep):
        for feature in evars:
            permuted[feature] = (
                df[feature].sample(frac=1, random_state=i).reset_index(drop=True)
            )
            importance_values[feature] = importance_values[feature] + imp_calc(
                baseline_fit, fitted.predict(permuted[evars])
            )
            permuted[feature] = df[feature]

    importance_values = {k: [v / rep] for k, v in importance_values.items()}
    sorted_idx = pd.DataFrame(importance_values).transpose()
    sorted_idx = sorted_idx.sort_values(0, ascending=True)
    fig = sorted_idx.plot.barh(color="slateblue", legend=None)
    plt.xlabel(xlab)
    plt.title("Permutation Importance")


def vimp_plot_sk(fitted, X, y, rep=5):
    """
    Creates permutation importance plots for models estimated using the
    sklearn library

    Parameters
    ----------
    fitted : A fitted sklearn objects
    X : Pandas DataFrame with data used for estimation
    rep: int
        The number of times to resample and calculate the permutation importance
    """

    if hasattr(fitted, "classes_"):
        scoring = "roc_auc"
        xlab = "Importance (AUC decrease)"
    else:
        scoring = "r2"
        xlab = "Importance (R-square decrease)"

    imp = permutation_importance(
        fitted, X, y, scoring=scoring, n_repeats=rep, random_state=1234
    )
    data = pd.DataFrame(imp.importances.T)
    data.columns = fitted.feature_names_in_
    order = data.agg("mean").sort_values(ascending=False).index
    fig = sns.barplot(
        x="value",
        y="variable",
        color="slateblue",
        errorbar=None,
        data=pd.melt(data[order]),
    )
    fig = fig.set(title="Permutation Importance", xlabel=xlab, ylabel=None)
