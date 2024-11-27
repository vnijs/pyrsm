import math
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import numpy as np
import seaborn as sns
import statsmodels as sm
from sklearn.inspection import permutation_importance
from pyrsm.stats import scale_df
from pyrsm.utils import ifelse, intersect, setdiff, check_dataframe, check_series
from .model import sim_prediction, extract_evars, extract_rvar, conditional_get_dummies
from .perf import auc


def distr_plot(
    data: pd.DataFrame | pl.DataFrame,
    cols: list = None,
    nint: int = 25,
    bins=25,
    **kwargs,
):
    """
    Plot histograms for numeric variables and frequency plots for categorical.
    variables. Columns of type integer with less than 25 unique values will be
    treated as categorical. To change this behavior, increase or decrease the
    value of the 'nint' argument

    Parameters
    ----------
    data : Pandas dataframe
    cols: A list of column names to generate distribution plots for. If None, all
        variables will be plotted
    nint: int
        The number of unique values in a series of type integer below which the
        series will be treated as a categorical variable
    bins: int
        The number of bins to use for histograms of numeric variables
    **kwargs : Named arguments to be passed to the pandas plotting methods
    """
    data = check_dataframe(data)
    if cols is None:
        all_cols = data.columns
        cols = []
        for i, c in enumerate(all_cols):
            if (
                not pd.api.types.is_numeric_dtype(data[c].dtype)
                and not pd.api.types.is_categorical_dtype(data[c].dtype)
            ) or pd.api.types.is_object_dtype(data[c].dtype):
                print(f"No plot will be created for {c} (type {data[c].dtype})")
            else:
                cols.append(c)

    data = data.loc[:, cols].copy()

    fig, axes = plt.subplots(max(math.ceil(data.shape[1] / 2), 2), 2, figsize=(10, 2 * max(data.shape[1], 4)))
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    row = 0
    for i, c in enumerate(data.columns):
        s = data[c]
        j = ifelse(i % 2 == 0, 0, 1)
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique() < nint:
            s.sort_values().value_counts(sort=False).plot.bar(
                ax=axes[row, j], title=c, rot=0, color="slateblue", **kwargs
            )
        elif pd.api.types.is_numeric_dtype(s.dtype):
            s.plot.hist(ax=axes[row, j], title=c, rot=0, color="slateblue", bins=bins, **kwargs)
        elif pd.api.types.is_categorical_dtype(s.dtype):
            s.value_counts(sort=False).plot.bar(ax=axes[row, j], title=c, rot=0, color="slateblue", **kwargs)
        else:
            print(f"No plot will be created for {c} (type {s.dtype})")

        if j == 1:
            row += 1

    # needed because the axes object must always be 2-dimensional
    # or else the indexing used in this function will fail for small
    # DataFrames
    if data.shape[1] < 3:
        axes[-1, -1].remove()
        axes[-1, 0].remove()
        if data.shape[1] == 1:
            axes[0, -1].remove()
    elif data.shape[1] % 2 == 1:
        axes[-1, -1].remove()

    # plt.show()


def pred_plot_sm(
    fitted,
    data: pd.DataFrame | pl.DataFrame,
    incl=None,
    excl=[],
    incl_int=[],
    fix=True,
    hline=False,
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
    data: Pandas or Polars DataFrame; dataset
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
    pred_plot(lr, data, excl="monetary")
    pred_plot(lr, data, incl = [], incl_int = ["frequency:monetary"])
    """
    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        # legacy for when only an un-fitted model was accepted
        model = fitted

    data = check_dataframe(data)

    # margin to add above and below in plots
    plot_margin = 0.025

    def calc_ylim(lab, lst, min_max):
        if isinstance(fix, bool) and fix:
            vals = lst[lab]
            min_vals = min(vals)
            max_vals = max(vals)
            mmin = min(min_max[0], min_vals - plot_margin * min_vals)
            mmax = max(min_max[1], max_vals + plot_margin * max_vals)
            return (mmin, mmax)
        elif not isinstance(fix, bool) and len(fix) == 2:
            return fix
        else:
            return False

    rvar = model.endog_names
    if isinstance(hline, bool):
        if hline:
            hline = data[rvar].mean()
            min_max = (hline - plot_margin * hline, hline + plot_margin * hline)
        else:
            hline = False
            min_max = (np.inf, -np.inf)
    else:
        min_max = (hline - plot_margin * hline, hline + plot_margin * hline)

    if incl is None:
        incl = extract_evars(model, data.columns)
    else:
        incl = ifelse(isinstance(incl, str), [incl], incl)

    excl = ifelse(isinstance(excl, str), [excl], excl)
    incl_int = ifelse(isinstance(incl_int, str), [incl_int], incl_int)

    if len(excl) > 0:
        incl = [i for i in incl if i not in excl]

    nr_plots = len(incl) + len(incl_int)
    if nr_plots == 0:
        return None
    fig, ax = plt.subplots(max(math.ceil(nr_plots / 2), 2), 2, figsize=(10, 2 * max(nr_plots, 4)))
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    pred_dict = {}
    for v in incl:
        iplot = sim_prediction(data, vary=v, nnv=nnv, minq=minq, maxq=maxq)
        iplot["prediction"] = fitted.predict(iplot)
        min_max = calc_ylim("prediction", iplot, min_max)
        pred_dict[v] = iplot

    for v in incl_int:
        vl = v.split(":")
        is_num = [pd.api.types.is_numeric_dtype(data[c].dtype) and data[c].nunique() > 5 for c in vl]
        iplot = sim_prediction(data, vary=vl, nnv=nnv, minq=minq, maxq=maxq)
        iplot["prediction"] = fitted.predict(iplot)
        if sum(is_num) < 2:
            min_max = calc_ylim("prediction", iplot, min_max)
        pred_dict[v] = iplot

    row = col = 0
    for i, v in enumerate(incl):
        col = ifelse(i % 2 == 0, 0, 1)
        if pd.api.types.is_numeric_dtype(data[v].dtype) and data[v].nunique() > 5:
            fig = sns.lineplot(x=v, y="prediction", data=pred_dict[v], ax=ax[row, col])
        else:
            fig = sns.lineplot(x=v, y="prediction", marker="o", data=pred_dict[v], ax=ax[row, col])
        if isinstance(min_max, tuple) and len(min_max) == 2:
            ax[row, col].set(ylim=min_max)
        if hline:
            ax[row, col].axhline(y=hline, linestyle="--")

        if col == 1:
            row += 1

    start_col = ifelse((col == 1) or len(incl) == 0, 0, 1)
    for j, v in enumerate(incl_int):
        col = ifelse(j % 2 == start_col, 0, 1)
        vl = v.split(":")
        is_num = [pd.api.types.is_numeric_dtype(data[c].dtype) and data[c].nunique() > 5 for c in vl]
        if sum(is_num) == 2:
            plot_data = (
                pred_dict[v]
                .pivot(index=vl[0], columns=vl[1], values="prediction")
                .transpose()
                .sort_values(vl[1], ascending=False)
            )
            fig = sns.heatmap(plot_data, ax=ax[row, col], xticklabels=False, yticklabels=False)
            # ax[i+j].imshow(plot_data) # not bad - investigate more
        elif sum(is_num) == 1:
            if is_num[1]:
                vl.reverse()
            fig = sns.lineplot(x=vl[0], y="prediction", hue=vl[1], data=pred_dict[v], ax=ax[row, col])
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
            ax[row, col].set(ylim=min_max)
        if sum(is_num) < 2 and hline:
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
    data: pd.DataFrame | pl.DataFrame,
    rvar=None,
    incl=None,
    excl=[],
    incl_int=[],
    transformed=None,
    fix=True,
    hline=False,
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
    data : Polars or Pandas DataFrame with data used for estimation. Should include categorical variables
        in their original form (i.e., before using get_dummies)
    rvar : The column name for the response/target variable
    incl : A list of column names to generate prediction plots for. If None, all
        variables will be plotted
    excl : A list of column names to exclude from plotting
    incl_int : A list is ":" separated column names to plots interaction plots for.
        For example incl_int = ["a:b", "b:c"] would generate interaction plots for
        variables a x b and b x c
    transformed : List of column names that were transformed using Pandas' get_dummies. If
        None, the function will try to determine which variables might have been transformed
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
    if hasattr(fitted, "feature_names_in_"):
        fn = fitted.feature_names_in_
    else:
        raise Exception(
            "This function requires a fitted sklearn model with a named features. If you are using one-hot encoding, please use get_dummies on your dataset before fitting the model."
        )

    if isinstance(data, dict):
        means = data["means"]
        stds = data["stds"]
        data = data["data"]
        # print(data.head())
        data = scale_df(data, means=means, stds=stds, sf=1)
        # print(data.head())
    else:
        means = None
        stds = None

    data = check_dataframe(data)

    not_transformed = [c for c in data.columns for f in fn if c == f]
    if transformed is None:
        transformed = list(
            set([c for c in data.columns for f in fn if f"{c}_" in f and c != f and f"{c}_" not in data.columns])
        )

    ints = intersect(transformed, not_transformed)
    if len(ints) > 0:
        trn = '", "'.join(transformed)
        ints = ", ".join(ints)
        not_transformed = setdiff(not_transformed, transformed)
        mess = f"""It is not clear which variables were (not) transformed using get_dummies. Please specify the
        transformed variables in your call to pred_plot_sk as `transformed=["{trn}"]`.
        In particular it is unclear how {ints} should be treated. Please remove entries from the list that
        are not correct and add any that are missing"""
        raise Exception(mess)

    if hasattr(fitted, "classes_"):
        sk_type = "classification"
    else:
        sk_type = "regression"

    def pred_fun(fitted, data):
        if sk_type == "classification":
            return fitted.predict_proba(data)[:, 1]
        else:
            if sk_type == "regression" and means is not None and stds is not None and rvar is not None:
                pred = fitted.predict(data) * stds[rvar] + means[rvar]
            else:
                pred = fitted.predict(data)

            return pred

    if incl is None:
        incl = not_transformed + transformed
    else:
        incl = ifelse(isinstance(incl, str), [incl], incl)

    def dummify(data, trs):
        if len(trs) > 0:
            return pd.concat([pd.get_dummies(data[trs], columns=trs), data.drop(trs, axis=1)], axis=1)
        else:
            return data

    # margin to add above and below in plots
    plot_margin = 0.025

    if isinstance(hline, bool):
        if hline and rvar is not None:
            hline = data[rvar].mean()
            min_max = (hline - plot_margin * hline, hline + plot_margin * hline)
        else:
            hline = False
            min_max = (np.inf, -np.inf)
    else:
        min_max = (hline - plot_margin * hline, hline + plot_margin * hline)

    def calc_ylim(lab, lst, min_max):
        if isinstance(fix, bool) and fix:
            vals = lst[lab]
            min_vals = min(vals)
            max_vals = max(vals)
            mmin = min(min_max[0], min_vals - plot_margin * min_vals)
            mmax = max(min_max[1], max_vals + plot_margin * max_vals)
            return (mmin, mmax)
        elif not isinstance(fix, bool) and len(fix) == 2:
            return fix
        else:
            return False

    excl = ifelse(isinstance(excl, str), [excl], excl)
    incl_int = ifelse(isinstance(incl_int, str), [incl_int], incl_int)

    if len(excl) > 0:
        incl = [i for i in incl if i not in excl]

    nr_plots = len(incl) + len(incl_int)
    fig, ax = plt.subplots(max(math.ceil(nr_plots / 2), 2), 2, figsize=(10, 2 * max(nr_plots, 4)))
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    pred_dict = {}
    for v in incl:
        iplot = sim_prediction(
            data[transformed + not_transformed].dropna(),
            vary=v,
            nnv=nnv,
            minq=minq,
            maxq=maxq,
        )
        iplot_dum = dummify(iplot, transformed)[fn]
        iplot["prediction"] = pred_fun(fitted, iplot_dum)
        min_max = calc_ylim("prediction", iplot, min_max)
        pred_dict[v] = iplot

    for v in incl_int:
        vl = v.split(":")
        is_num = [pd.api.types.is_numeric_dtype(data[c].dtype) and data[c].nunique() > 5 for c in vl]
        iplot = sim_prediction(
            data[transformed + not_transformed].dropna(),
            vary=vl,
            nnv=nnv,
            minq=minq,
            maxq=maxq,
        )
        iplot_dum = dummify(iplot, transformed)[fn]
        iplot["prediction"] = pred_fun(fitted, iplot_dum)
        if sum(is_num) < 2:
            min_max = calc_ylim("prediction", iplot, min_max)
        pred_dict[v] = iplot

    if len(pred_dict) > 0 and means is not None and stds is not None:
        for k in pred_dict.keys():
            for c in means.keys():
                if c in pred_dict[k].columns:
                    pred_dict[k][c] = pred_dict[k][c] * stds[c] + means[c]

    row = col = 0
    for i, v in enumerate(incl):
        col = ifelse(i % 2 == 0, 0, 1)
        if pd.api.types.is_numeric_dtype(data[v].dtype) and data[v].nunique() > 5:
            fig = sns.lineplot(x=v, y="prediction", data=pred_dict[v], ax=ax[row, col])
        else:
            fig = sns.lineplot(x=v, y="prediction", marker="o", data=pred_dict[v], ax=ax[row, col])
        if isinstance(min_max, tuple) and len(min_max) == 2:
            ax[row, col].set(ylim=min_max)
        if hline:
            ax[row, col].axhline(y=hline, linestyle="--")

        if col == 1:
            row += 1

    start_col = ifelse((col == 1) or len(incl) == 0, 0, 1)
    for j, v in enumerate(incl_int):
        col = ifelse(j % 2 == start_col, 0, 1)
        vl = v.split(":")
        is_num = [pd.api.types.is_numeric_dtype(data[c].dtype) and data[c].nunique() > 5 for c in vl]
        if sum(is_num) == 2:
            plot_data = (
                pred_dict[v]
                .pivot(index=vl[0], columns=vl[1], values="prediction")
                .transpose()
                .sort_values(vl[1], ascending=False)
            )
            fig = sns.heatmap(plot_data, ax=ax[row, col], xticklabels=False, yticklabels=False)
            # ax[i+j].imshow(plot_data) # not bad - investigate more
        elif sum(is_num) == 1:
            if is_num[1]:
                vl.reverse()
            fig = sns.lineplot(x=vl[0], y="prediction", hue=vl[1], data=pred_dict[v], ax=ax[row, col])
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
            ax[row, col].set(ylim=min_max)
        if sum(is_num) < 2 and hline:
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


def vimp_plot_sm(fitted, data, rep=10, ax=None, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    statsmodels library

    Parameters
    ----------
    fitted : A fitted statsmodels objects
    data : Polars or Pandas DataFrame with data used for estimation
    rep: int
        The number of times to resample and calculate the permutation importance
    ax : axis object
        Add the plot to a the axis object from a matplotlib subplot
    ret: bool
        Return the variable importance table as a sorted DataFrame
    """
    fw = None
    if hasattr(fitted, "model"):
        model = fitted.model
        if hasattr(model, "_has_freq_weights") and model._has_freq_weights:
            fw = model.freq_weights
    else:
        return "This function requires a fitted linear or logistic regression"

    rvar = extract_rvar(model, data.columns)
    evars = extract_evars(model, data.columns)
    data = data[[rvar] + evars].copy().dropna().reset_index(drop=True)

    if len(model.endog) != data.shape[0]:
        raise Exception(
            "The number of rows in the DataFrame should be the same as the number of rows in the data used to estimate the model"
        )

    def imp_calc_reg(base, pred):
        return base - pd.DataFrame({"y": model.endog, "yhat": pred}).corr().iloc[0, 1] ** 2

    def imp_calc_logit(base, pred):
        return base - auc(model.endog, pred, weights=fw)

    # Calculate the baseline performance
    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        baseline_fit = auc(model.endog, fitted.predict(data[evars]), weights=fw)
        imp_calc = imp_calc_logit  # specifying the function to use
        xlab = "Importance (AUC decrease)"
    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        baseline_fit = pd.DataFrame({"y": model.endog, "yhat": fitted.predict(data[evars])}).corr().iloc[0, 1] ** 2
        imp_calc = imp_calc_reg  # specifying the function to use
        xlab = "Importance (R-square decrease)"
    else:
        return "This model type is not supported. For sklearn models use vimp_plot_sk"

    # Create a copy of the dataframe
    permuted = data.copy()

    # Initialize a dictionary to store the permutation importance values
    importance_values = {v: 0 for v in evars}

    # Iterate over each feature
    for i in range(rep):
        for feature in evars:
            permuted[feature] = data[feature].sample(frac=1, random_state=i).reset_index(drop=True)
            importance_values[feature] = importance_values[feature] + imp_calc(
                baseline_fit, fitted.predict(permuted[evars])
            )
            permuted[feature] = data[feature]

    importance_values = {k: [v / rep] for k, v in importance_values.items()}
    sorted_idx = pd.DataFrame(importance_values).transpose()
    sorted_idx = sorted_idx.sort_values(0, ascending=True)
    print(importance_values)
    fig = sorted_idx.plot.barh(
        color="slateblue",
        legend=None,
        figsize=(6, max(5, len(sorted_idx) * 0.4)),
        ax=ax,
    )
    plt.xlabel(xlab)
    plt.title("Permutation Importance")

    if ret:
        sorted_idx.columns = ["Importance"]
        return sorted_idx[::-1]


def vimp_plot_sk(fitted, X, y, rep=5, ax=None, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    sklearn library

    Parameters
    ----------
    fitted : A fitted sklearn objects
    X : Polars or Pandas DataFrame with data containing the explanatory variables (features) used for estimation
    y : Series with data for the response variable (target) used for estimation
    rep: int
        The number of times to resample and calculate permutation importance
    ax : axis object
        Add the plot to a the axis object from a matplotlib subplot
    ret: bool
        Return the variable importance table as a sorted DataFrame
    """

    if hasattr(fitted, "classes_"):
        scoring = "roc_auc"
        xlab = "Importance (AUC decrease)"
    else:
        scoring = "r2"
        xlab = "Importance (R-square decrease)"

    # convert to (copied) pandas objects as needed
    X = check_dataframe(X)
    y = check_series(y)

    imp = permutation_importance(fitted, X, y, scoring=scoring, n_repeats=rep, random_state=1234)
    data = pd.DataFrame(imp.importances.T)
    # print(data)
    # print(fitted.feature_names_in_)
    data.columns = fitted.feature_names_in_
    sorted_idx = pd.DataFrame(data.mean().sort_values())
    fig = sorted_idx.plot.barh(
        color="slateblue",
        legend=None,
        figsize=(6, max(5, len(sorted_idx) * 0.4)),
        ax=ax,
    )
    fig = fig.set(title="Permutation Importance", xlabel=xlab, ylabel=None)

    if ret:
        sorted_idx.columns = ["Importance"]
        return sorted_idx[::-1]


def vimp_plot_sk2(model, rep=10, ax=None, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    sklearn library

    Parameters
    ----------
    fitted : A fitted sklearn objects
    rep: int
        The number of times to resample and calculate permutation importance
    ax : axis object
        Add the plot to a the axis object from a matplotlib subplot
    ret: bool
        Return the variable importance table as a sorted DataFrame
    """
    # if hasattr(fitted, "model"):
    #     model = fitted.model
    #     if hasattr(model, "_has_freq_weights") and model._has_freq_weights:
    #         fw = model.freq_weights
    # else:
    #     return "This function requires a fitted linear or logistic regression"

    rvar = model.rvar
    evars = model.evar
    data = model.data[[rvar] + evars].copy().reset_index(drop=True).dropna()

    def imp_calc_reg(base, pred):
        return base - pd.DataFrame({"y": data[rvar], "yhat": pred}).corr().iloc[0, 1] ** 2

    def imp_calc_clf(base, pred):
        # base = 0.751
        pauc = auc(data[rvar], pred)
        print(base, pauc, base - pred)
        return base - auc(data[rvar], pred)

    # Calculate the baseline performance
    if hasattr(model.fitted, "classes_"):
        xlab = "Importance (AUC decrease)"
        baseline_fit = auc(data[rvar], model.fitted.predict(model.data_onehot))
        imp_calc = imp_calc_clf  # specifying the function to use
    else:
        baseline_fit = (
            pd.DataFrame({"y": data[rvar], "yhat": model.fitted.predict(model.data_onehot)}).corr().iloc[0, 1] ** 2
        )
        imp_calc = imp_calc_reg  # specifying the function to use
        xlab = "Importance (R-square decrease)"

    # Create a copy of the dataframe
    permuted = data.copy()

    # Initialize a dictionary to store the permutation importance values
    importance_values = {v: 0 for v in evars}
    print(baseline_fit)

    # Iterate over each feature
    for i in range(rep):
        for feature in evars:
            permuted[feature] = data[feature].sample(frac=1, random_state=i).reset_index(drop=True)
            importance_values[feature] += imp_calc(
                baseline_fit, model.fitted.predict(conditional_get_dummies(permuted[evars]))
            )
            permuted[feature] = data[feature]

    importance_values = {k: [v / rep] for k, v in importance_values.items()}
    sorted_idx = pd.DataFrame(importance_values).transpose()
    sorted_idx = sorted_idx.sort_values(0, ascending=True)
    fig = sorted_idx.plot.barh(
        color="slateblue",
        legend=None,
        figsize=(6, max(5, len(sorted_idx) * 0.4)),
        ax=ax,
    )
    plt.xlabel(xlab)
    plt.title("Permutation Importance")

    if ret:
        sorted_idx.columns = ["Importance"]
        return sorted_idx[::-1]
