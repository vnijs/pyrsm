import math
import time

import numpy as np
import pandas as pd
import polars as pl
import statsmodels as sm
from plotnine import (
    aes,
    coord_flip,
    element_text,
    geom_bar,
    geom_histogram,
    geom_hline,
    geom_line,
    geom_point,
    geom_tile,
    ggplot,
    ggtitle,
    labs,
    scale_fill_gradient2,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_bw,
)
from sklearn.inspection import permutation_importance

from pyrsm.stats import scale_df
from pyrsm.utils import ifelse, intersect, setdiff, expand_grid

from .model import (
    extract_evars,
    extract_rvar,
    get_dummies,
    sim_prediction,
    to_pandas_with_categories,
)
from .perf import auc


def _is_numeric_with_many_unique(data: pl.DataFrame, col: str, threshold: int = 5) -> bool:
    """Check if column is numeric with more than threshold unique values (polars version)."""
    dtype = data.schema.get(col)
    if dtype is None:
        return False
    if not dtype.is_numeric():
        return False
    n_unique = data.select(pl.col(col).n_unique()).item()
    return n_unique > threshold


def _is_categorical(df: pl.DataFrame, col: str, nint: int = 25) -> bool:
    """Check if column should be treated as categorical."""
    dtype = df.schema.get(col)
    if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum):
        return True
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        n_unique = df.select(pl.col(col).n_unique()).item()
        return n_unique < nint
    return False


def _calc_r_squared(y, yhat) -> float:
    """Calculate R² using polars correlation."""
    # Handle both Series and lists
    if isinstance(y, pl.Series):
        y_vals = y
    else:
        y_vals = pl.Series("y", list(y))

    if isinstance(yhat, pl.Series):
        yhat_vals = yhat
    else:
        yhat_vals = pl.Series("yhat", list(yhat))

    corr = pl.DataFrame({"y": y_vals, "yhat": yhat_vals}).select(pl.corr("y", "yhat")).item()
    return corr**2 if corr is not None else 0.0


def _compose_plots(plot_list: list, ncol: int = 2):
    """Compose a list of plots into a grid using plotnine's | and / operators."""
    if len(plot_list) == 0:
        return None
    if len(plot_list) == 1:
        return plot_list[0]

    nrow = math.ceil(len(plot_list) / ncol)

    # Build rows (side by side with |)
    rows = []
    for i in range(nrow):
        start_idx = i * ncol
        end_idx = min(start_idx + ncol, len(plot_list))
        row_plots = plot_list[start_idx:end_idx]

        if len(row_plots) == 1:
            rows.append(row_plots[0])
        else:
            row = row_plots[0]
            for p in row_plots[1:]:
                row = row | p
            rows.append(row)

    # Stack rows vertically with /
    if len(rows) == 1:
        combined = rows[0]
    else:
        combined = rows[0]
        for row in rows[1:]:
            combined = combined / row

    # Auto-adjust figure size
    height_per_row = 3
    width_per_col = 4
    fig_width = width_per_col * min(ncol, len(plot_list))
    fig_height = height_per_row * nrow
    combined = combined + theme(figure_size=(fig_width, fig_height))

    return combined


def distr_plot(
    data: pl.DataFrame | pd.DataFrame,
    cols: list = None,
    nint: int = 25,
    bins: int = 25,
    ncol: int = 2,
):
    """
    Plot histograms for numeric variables and frequency plots for categorical
    variables using plotnine. Columns of type integer with less than nint unique
    values will be treated as categorical.

    Parameters
    ----------
    data : pl.DataFrame or pd.DataFrame
        Input data
    cols : list, optional
        Column names to plot. If None, all plottable columns are used
    nint : int, default 25
        Number of unique values below which integers are treated as categorical
    bins : int, default 25
        Number of bins for histograms
    ncol : int, default 2
        Number of columns in the plot grid

    Returns
    -------
    plotnine.ggplot or plotnine.composition.Compose
        Single plot or combined plot composition
    """
    # Convert pandas to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    # Determine which columns to plot
    if cols is None:
        cols = []
        for c in data.columns:
            dtype = data.schema.get(c)
            if dtype.is_numeric() or dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum):
                cols.append(c)
            else:
                print(f"No plot will be created for {c} (type {dtype})")

    if not cols:
        return None

    # Create individual plots for each column
    plot_list = []
    for c in cols:
        is_cat = _is_categorical(data, c, nint)

        if is_cat:
            # Bar plot for categorical
            col_data = data.select(pl.col(c).cast(pl.Utf8).alias("value"))
            p = (
                ggplot(col_data, aes(x="value"))
                + geom_bar(fill="slateblue", alpha=0.8)
                + labs(x="", y="Count")
                + ggtitle(c)
                + theme_bw()
                + theme(
                    plot_title=element_text(size=10, weight="bold"),
                    axis_text_x=element_text(rotation=45, ha="right"),
                )
            )
        else:
            # Histogram for numeric
            col_data = data.select(pl.col(c).cast(pl.Float64).alias("value"))
            p = (
                ggplot(col_data, aes(x="value"))
                + geom_histogram(bins=bins, fill="slateblue", alpha=0.8)
                + labs(x="", y="Count")
                + ggtitle(c)
                + theme_bw()
                + theme(plot_title=element_text(size=10, weight="bold"))
            )

        plot_list.append(p)

    return _compose_plots(plot_list, ncol)


def pred_plot_sm(
    fitted,
    data: pl.DataFrame | pd.DataFrame,
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
    data: Polars or Pandas DataFrame; dataset
    incl: List of strings; contains the names of the columns of data to use for prediction
          By default it will extract the names of all explanatory variables used in estimation
          Use [] to ensure no single-variable plots are created
    excl: List of strings; contains names of columns to exclude
    incl_int: List of strings; contains the names of the columns of data to be interacted for
          prediction plotting (e.g., ["x1:x2", "x2:x3"])
    fix : Logical or tuple
        Set the desired limited on yhat or have it calculated automatically.
        Set to FALSE to have y-axis limits set for each plot
    hline : Logical or float
        Add a horizontal line at the average of the target variable
    nnv: Integer: The number of values to simulate for numeric variables
    minq : float
        Quantile to use for the minimum value for simulation of numeric variables
    maxq : float
        Quantile to use for the maximum value for simulation of numeric variables

    Returns
    -------
    plotnine plot composition
    """
    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        model = fitted

    # Convert pandas to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    # margin to add above and below in plots
    plot_margin = 0.025

    def calc_ylim(vals, min_max):
        if isinstance(fix, bool) and fix:
            min_vals = min(vals)
            max_vals = max(vals)
            mmin = min(min_max[0], min_vals - plot_margin * abs(min_vals))
            mmax = max(min_max[1], max_vals + plot_margin * abs(max_vals))
            return (mmin, mmax)
        elif not isinstance(fix, bool) and len(fix) == 2:
            return fix
        else:
            return min_max

    rvar = model.endog_names
    if isinstance(hline, bool):
        if hline:
            hline_val = data.select(pl.col(rvar).mean()).item()
            min_max = (
                hline_val - plot_margin * abs(hline_val),
                hline_val + plot_margin * abs(hline_val),
            )
        else:
            hline_val = None
            min_max = (float("inf"), float("-inf"))
    else:
        hline_val = hline
        min_max = (hline - plot_margin * abs(hline), hline + plot_margin * abs(hline))

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

    # Generate predictions for each variable
    pred_dict = {}
    for v in incl:
        iplot = sim_prediction(data, vary=v, nnv=nnv, minq=minq, maxq=maxq)
        # statsmodels needs pandas with proper categories for prediction
        iplot_pd = to_pandas_with_categories(iplot, data)
        predictions = fitted.predict(iplot_pd)
        iplot = iplot.with_columns(prediction=pl.Series(predictions))
        min_max = calc_ylim(predictions, min_max)
        pred_dict[v] = iplot

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        iplot = sim_prediction(data, vary=vl, nnv=nnv, minq=minq, maxq=maxq)
        iplot_pd = to_pandas_with_categories(iplot, data)
        predictions = fitted.predict(iplot_pd)
        iplot = iplot.with_columns(prediction=pl.Series(predictions))
        if sum(is_num) < 2:
            min_max = calc_ylim(predictions, min_max)
        pred_dict[v] = iplot

    # Create plots
    plot_list = []
    for v in incl:
        plot_data = pred_dict[v]
        is_num = _is_numeric_with_many_unique(data, v, 5)

        if is_num:
            p = ggplot(plot_data, aes(x=v, y="prediction")) + geom_line(color="steelblue")
        else:
            p = (
                ggplot(plot_data, aes(x=v, y="prediction"))
                + geom_line(color="steelblue", group=1)
                + geom_point(color="steelblue", size=3)
            )

        p = p + labs(x="", y="Prediction") + ggtitle(v) + theme_bw()

        if isinstance(min_max, tuple) and min_max[0] != float("inf"):
            p = p + scale_y_continuous(limits=min_max)

        if hline_val is not None:
            p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

        plot_list.append(p)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        plot_data = pred_dict[v]

        if sum(is_num) == 2:
            # Heatmap for two numeric variables
            p = (
                ggplot(plot_data, aes(x=vl[0], y=vl[1], fill="prediction"))
                + geom_tile()
                + scale_fill_gradient2(low="blue", mid="white", high="red")
                + labs(x=vl[0], y=vl[1], fill="Prediction")
                + ggtitle(v)
                + theme_bw()
            )
        elif sum(is_num) == 1:
            # Line plot with color grouping
            if is_num[1]:
                vl = [vl[1], vl[0]]
            plot_data = plot_data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line()
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
            if isinstance(min_max, tuple) and min_max[0] != float("inf"):
                p = p + scale_y_continuous(limits=min_max)
            if hline_val is not None:
                p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")
        else:
            # Both categorical
            plot_data = plot_data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line(aes(group=vl[1]))
                + geom_point(size=3)
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
            if isinstance(min_max, tuple) and min_max[0] != float("inf"):
                p = p + scale_y_continuous(limits=min_max)
            if hline_val is not None:
                p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

        plot_list.append(p)

    return _compose_plots(plot_list, ncol=2)


def pred_plot_sk(
    fitted,
    data: pl.DataFrame | pd.DataFrame,
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
    ret=False,
):
    """
    Generate prediction plots for sklearn models. A faster alternative to PDP plots
    that can handle interaction plots with categorical variables.

    Parameters
    ----------
    fitted : A fitted sklearn model
    data : Polars DataFrame with data used for estimation
    rvar : The column name for the response/target variable
    incl : A list of column names to generate prediction plots for
    excl : A list of column names to exclude from plotting
    incl_int : A list of ":" separated column names for interaction plots
    transformed : List of column names that were transformed using get_dummies
    fix : Logical or tuple for y-axis limits
    hline : Logical or float for horizontal line
    nnv: int - number of values for numeric variables
    minq : float - quantile for minimum value
    maxq : float - quantile for maximum value
    ret : Return the prediciton dictionary for testing purposes

    Returns
    -------
    plotnine plot composition
    """
    # features names used in the sklearn model
    if hasattr(fitted, "feature_names_in_"):
        fn = fitted.feature_names_in_
    else:
        raise Exception("This function requires a fitted sklearn model with named features.")

    if isinstance(data, dict):
        means = data["means"]
        stds = data["stds"]
        data = data["data"]
        # Convert pandas to polars if needed
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        data_scaled = scale_df(data.to_pandas(), means=means, stds=stds, sf=1)
        data_scaled = pl.from_pandas(data_scaled)
    else:
        means = None
        stds = None
        data_scaled = None
        # Convert pandas to polars if needed
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

    not_transformed = [c for c in data.columns for f in fn if c == f]
    if transformed is None:
        transformed = list(
            set(
                [
                    c
                    for c in data.columns
                    for f in fn
                    if f"{c}_" in f and c != f and f"{c}_" not in data.columns
                ]
            )
        )

    ints = intersect(transformed, not_transformed)
    if len(ints) > 0:
        trn = '", "'.join(transformed)
        ints = ", ".join(ints)
        not_transformed = setdiff(not_transformed, transformed)
        mess = f"""It is not clear which variables were (not) transformed using get_dummies. Please specify the
        transformed variables in your call to pred_plot_sk as `transformed=["{trn}"]`."""
        raise Exception(mess)

    if hasattr(fitted, "classes_"):
        sk_type = "classification"
    else:
        sk_type = "regression"

    def pred_fun(fitted, data_for_pred):
        data_pd = (
            data_for_pred.to_pandas() if isinstance(data_for_pred, pl.DataFrame) else data_for_pred
        )
        if sk_type == "classification":
            return fitted.predict_proba(data_pd)[:, 1]
        else:
            if means is not None and stds is not None and rvar is not None:
                pred = fitted.predict(data_pd) * stds[rvar] + means[rvar]
            else:
                pred = fitted.predict(data_pd)
            return pred

    if incl is None:
        incl = not_transformed + transformed
    else:
        incl = ifelse(isinstance(incl, str), [incl], incl)

    def dummify(data_to_dum, trs):
        if len(trs) > 0:
            categories = {}
            for col in trs:
                prefix = f"{col}_"
                categories[col] = [f.replace(prefix, "") for f in fn if f.startswith(prefix)]

            result = get_dummies(
                data_to_dum.select(trs),
                drop_first=False,
                drop_nonvarying=False,
                categories=categories,
            )

            non_trs = [c for c in data_to_dum.columns if c not in trs]
            if non_trs:
                result = pl.concat([result, data_to_dum.select(non_trs)], how="horizontal")

            return result
        else:
            return data_to_dum

    # margin to add above and below in plots
    plot_margin = 0.025

    if isinstance(hline, bool):
        if hline and rvar is not None:
            hline_val = data.select(pl.col(rvar).mean()).item()
            min_max = (
                hline_val - plot_margin * abs(hline_val),
                hline_val + plot_margin * abs(hline_val),
            )
        else:
            hline_val = None
            min_max = (float("inf"), float("-inf"))
    else:
        hline_val = hline
        min_max = (hline - plot_margin * abs(hline), hline + plot_margin * abs(hline))

    def calc_ylim(vals, min_max):
        if isinstance(fix, bool) and fix:
            min_vals = min(vals)
            max_vals = max(vals)
            mmin = min(min_max[0], min_vals - plot_margin * abs(min_vals))
            mmax = max(min_max[1], max_vals + plot_margin * abs(max_vals))
            return (mmin, mmax)
        elif not isinstance(fix, bool) and len(fix) == 2:
            return fix
        else:
            return min_max

    excl = ifelse(isinstance(excl, str), [excl], excl)
    incl_int = ifelse(isinstance(incl_int, str), [incl_int], incl_int)

    if len(excl) > 0:
        incl = [i for i in incl if i not in excl]

    nr_plots = len(incl) + len(incl_int)
    if nr_plots == 0:
        return None

    # Generate predictions
    pred_dict = {}
    for v in incl:
        iplot = sim_prediction(
            data.select(transformed + not_transformed).drop_nulls(),
            vary=v,
            nnv=nnv,
            minq=minq,
            maxq=maxq,
        )
        iplot_dum = dummify(iplot, transformed).select(fn)

        # Scale numeric columns before prediction if means/stds provided
        if means is not None and stds is not None:
            for col in not_transformed:
                if col in means and col in iplot_dum.columns:
                    iplot_dum = iplot_dum.with_columns(
                        ((pl.col(col) - means[col]) / stds[col]).alias(col)
                    )

        predictions = pred_fun(fitted, iplot_dum)
        iplot = iplot.with_columns(prediction=pl.Series(predictions))
        min_max = calc_ylim(predictions, min_max)
        pred_dict[v] = iplot

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        iplot = sim_prediction(
            data.select(transformed + not_transformed).drop_nulls(),
            vary=vl,
            nnv=nnv,
            minq=minq,
            maxq=maxq,
        )
        iplot_dum = dummify(iplot, transformed).select(fn)

        # Scale numeric columns before prediction if means/stds provided
        if means is not None and stds is not None:
            for col in not_transformed:
                if col in means and col in iplot_dum.columns:
                    iplot_dum = iplot_dum.with_columns(
                        ((pl.col(col) - means[col]) / stds[col]).alias(col)
                    )

        predictions = pred_fun(fitted, iplot_dum)
        iplot = iplot.with_columns(prediction=pl.Series(predictions))
        if sum(is_num) < 2:
            min_max = calc_ylim(predictions, min_max)
        pred_dict[v] = iplot

    if len(pred_dict) > 0 and means is not None and stds is not None:
        for k in pred_dict.keys():
            for c in means.keys():
                if c in pred_dict[k].columns:
                    pred_dict[k] = pred_dict[k].with_columns(pl.col(c) * stds[c] + means[c])

    if ret:
        return pred_dict

    # Create plots
    plot_list = []
    for v in incl:
        plot_data = pred_dict[v]
        is_num = _is_numeric_with_many_unique(data, v, 5)

        if is_num:
            p = ggplot(plot_data, aes(x=v, y="prediction")) + geom_line(color="steelblue")
        else:
            p = (
                ggplot(plot_data, aes(x=v, y="prediction"))
                + geom_line(color="steelblue", group=1)
                + geom_point(color="steelblue", size=3)
            )

        p = p + labs(x="", y="Prediction") + ggtitle(v) + theme_bw()

        if isinstance(min_max, tuple) and min_max[0] != float("inf"):
            p = p + scale_y_continuous(limits=min_max)

        if hline_val is not None:
            p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

        plot_list.append(p)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        plot_data = pred_dict[v]

        if sum(is_num) == 2:
            p = (
                ggplot(plot_data, aes(x=vl[0], y=vl[1], fill="prediction"))
                + geom_tile()
                + scale_fill_gradient2(low="blue", mid="white", high="red")
                + labs(x=vl[0], y=vl[1], fill="Prediction")
                + ggtitle(v)
                + theme_bw()
            )
        elif sum(is_num) == 1:
            if is_num[1]:
                vl = [vl[1], vl[0]]
            plot_data = plot_data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line()
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
            if isinstance(min_max, tuple) and min_max[0] != float("inf"):
                p = p + scale_y_continuous(limits=min_max)
            if hline_val is not None:
                p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")
        else:
            plot_data = plot_data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line(aes(group=vl[1]))
                + geom_point(size=3)
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
            if isinstance(min_max, tuple) and min_max[0] != float("inf"):
                p = p + scale_y_continuous(limits=min_max)
            if hline_val is not None:
                p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

        plot_list.append(p)

    return _compose_plots(plot_list, ncol=2)


def pdp_sk(
    fitted,
    data: pl.DataFrame | pd.DataFrame,
    rvar=None,
    incl=None,
    excl=[],
    incl_int=[],
    transformed=None,
    mode="pdp",
    n_sample=2000,
    grid_resolution=50,
    minq=0.05,
    maxq=0.95,
    interaction_slices=5,
    fix=True,
    hline=True,
    ncol=2,
):
    """
    Generate Partial Dependence Plots (PDP) for sklearn models.

    Parameters
    ----------
    fitted : A fitted sklearn model
    data : Polars or Pandas DataFrame with data used for estimation
    rvar : The column name for the response/target variable
    incl : A list of column names to generate PDP plots for
    excl : A list of column names to exclude from plotting
    incl_int : A list of ":" separated column names for interaction plots
    transformed : List of column names that were transformed using get_dummies
    mode : str, "fast" or "pdp"
        "fast" - uses sim_prediction (mean/mode for other vars, like pred_plot_sk)
        "pdp" - true PDP: samples rows, replaces feature values, averages predictions
    n_sample : int
        Number of samples to use for PDP mode (caps at dataset size)
    grid_resolution : int
        Number of grid points for numeric variables
    minq : float
        Quantile for minimum value of numeric variables
    maxq : float
        Quantile for maximum value of numeric variables
    interaction_slices : int
        Number of slices for numeric-numeric interactions (line plot)
    fix : Logical or tuple for y-axis limits
    hline : Logical or float for horizontal line
    ncol : int
        Number of columns in plot grid

    Returns
    -------
    tuple: (plot, data_dict, runtime_seconds)
        - plot: plotnine plot composition
        - data_dict: dict of DataFrames with underlying PDP data
        - runtime_seconds: total computation time
    """
    start_time = time.time()

    # Convert to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    # Get feature names from sklearn model
    if hasattr(fitted, "feature_names_in_"):
        fn = fitted.feature_names_in_
    else:
        raise ValueError("This function requires a fitted sklearn model with named features.")

    # Determine model type
    if hasattr(fitted, "classes_"):
        sk_type = "classification"
    else:
        sk_type = "regression"

    # Identify transformed and not-transformed variables
    not_transformed = [c for c in data.columns for f in fn if c == f]
    if transformed is None:
        transformed = list(
            set(
                [
                    c
                    for c in data.columns
                    for f in fn
                    if f"{c}_" in f and c != f and f"{c}_" not in data.columns
                ]
            )
        )

    ints = intersect(transformed, not_transformed)
    if len(ints) > 0:
        trn = '", "'.join(transformed)
        raise ValueError(
            f'Unclear which variables were transformed. Please specify transformed=["{trn}"].'
        )

    # Build include list
    if incl is None:
        incl = not_transformed + transformed
    else:
        incl = ifelse(isinstance(incl, str), [incl], incl)

    excl = ifelse(isinstance(excl, str), [excl], excl)
    incl_int = ifelse(isinstance(incl_int, str), [incl_int], incl_int)

    if len(excl) > 0:
        incl = [i for i in incl if i not in excl]

    nr_plots = len(incl) + len(incl_int)
    if nr_plots == 0:
        return None, {}, 0.0

    # Helper to create dummy columns
    def dummify(data_to_dum, trs):
        if len(trs) > 0:
            categories = {}
            for col in trs:
                prefix = f"{col}_"
                categories[col] = [f.replace(prefix, "") for f in fn if f.startswith(prefix)]

            result = get_dummies(
                data_to_dum.select(trs),
                drop_first=False,
                drop_nonvarying=False,
                categories=categories,
            )

            non_trs = [c for c in data_to_dum.columns if c not in trs]
            if non_trs:
                result = pl.concat([result, data_to_dum.select(non_trs)], how="horizontal")
            return result
        else:
            return data_to_dum

    # Prediction function
    def pred_fun(fitted, data_for_pred):
        data_pd = (
            data_for_pred.to_pandas() if isinstance(data_for_pred, pl.DataFrame) else data_for_pred
        )
        if sk_type == "classification":
            return fitted.predict_proba(data_pd)[:, 1]
        else:
            return fitted.predict(data_pd)

    # Get base data for predictions
    base_data = data.select(transformed + not_transformed).drop_nulls()

    # Sample data for PDP mode
    n_obs = base_data.height
    if mode == "pdp":
        sample_size = min(n_sample, n_obs)
        if sample_size < n_obs:
            sample_data = base_data.sample(sample_size, seed=1234)
        else:
            sample_data = base_data
    else:
        sample_data = base_data

    # Helper to build grid for a variable
    def build_grid(var):
        col = data[var]
        if _is_categorical(data, var, 5):
            return col.unique().drop_nulls().to_list()
        else:
            nu = col.n_unique()
            min_val = col.quantile(minq)
            max_val = col.quantile(maxq)
            return np.linspace(min_val, max_val, min(nu, grid_resolution)).tolist()

    # Compute PDP for single variable
    def compute_pdp_single(var):
        grid_vals = build_grid(var)
        predictions = []

        if mode == "pdp":
            # True PDP: for each grid value, replace var in sample, predict, average
            for gv in grid_vals:
                modified = sample_data.with_columns(
                    pl.lit(gv).cast(sample_data[var].dtype).alias(var)
                )
                modified_dum = dummify(modified, transformed).select(fn)
                preds = pred_fun(fitted, modified_dum)
                predictions.append(np.mean(preds))
        else:
            # Fast mode: use sim_prediction style (other vars at mean/mode)
            iplot = sim_prediction(base_data, vary=var, nnv=grid_resolution, minq=minq, maxq=maxq)
            iplot_dum = dummify(iplot, transformed).select(fn)
            preds = pred_fun(fitted, iplot_dum)
            grid_vals = iplot[var].to_list()
            predictions = preds.tolist()

        return pl.DataFrame({var: grid_vals, "prediction": predictions})

    # Compute PDP for interaction (two variables)
    def compute_pdp_interaction(var1, var2):
        is_num1 = _is_numeric_with_many_unique(data, var1, 5)
        is_num2 = _is_numeric_with_many_unique(data, var2, 5)

        grid1 = build_grid(var1)
        grid2 = build_grid(var2)

        # For num-num, use slices for line plot (not full grid)
        if is_num1 and is_num2:
            # Slice var2 into interaction_slices levels
            grid2 = np.linspace(
                data[var2].quantile(minq),
                data[var2].quantile(maxq),
                interaction_slices,
            ).tolist()

        # Create full grid
        schema = {var1: data[var1].dtype, var2: data[var2].dtype}
        grid_df = expand_grid({var1: grid1, var2: grid2}, schema)
        predictions = []

        if mode == "pdp":
            # For each grid point, replace both vars in sample, predict, average
            for row in grid_df.iter_rows(named=True):
                modified = sample_data.with_columns(
                    pl.lit(row[var1]).cast(sample_data[var1].dtype).alias(var1),
                    pl.lit(row[var2]).cast(sample_data[var2].dtype).alias(var2),
                )
                modified_dum = dummify(modified, transformed).select(fn)
                preds = pred_fun(fitted, modified_dum)
                predictions.append(np.mean(preds))
        else:
            # Fast mode
            for row in grid_df.iter_rows(named=True):
                iplot = sim_prediction(base_data, vary={var1: [row[var1]], var2: [row[var2]]})
                iplot_dum = dummify(iplot, transformed).select(fn)
                preds = pred_fun(fitted, iplot_dum)
                predictions.append(preds[0])

        return grid_df.with_columns(prediction=pl.Series(predictions))

    # Plot margin and y-axis limits
    plot_margin = 0.025

    if isinstance(hline, bool):
        if hline and rvar is not None and rvar in data.columns:
            col = data[rvar]
            # For numeric columns, use mean; for categorical, compute proportion
            if col.dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                hline_val = col.mean()
            else:
                # Categorical: compute proportion of positive class
                # Use type-appropriate values for comparison
                if col.dtype == pl.Boolean:
                    hline_val = col.mean()
                else:
                    # String/categorical: check for common positive labels
                    positive_str = ["Yes", "yes", "YES", "1", "True", "true"]
                    hline_val = col.cast(pl.Utf8).is_in(positive_str).mean()

            if hline_val is not None:
                min_max = (
                    hline_val - plot_margin * abs(hline_val),
                    hline_val + plot_margin * abs(hline_val),
                )
            else:
                min_max = (float("inf"), float("-inf"))
        else:
            hline_val = None
            min_max = (float("inf"), float("-inf"))
    else:
        hline_val = hline
        min_max = (hline - plot_margin * abs(hline), hline + plot_margin * abs(hline))

    def calc_ylim(vals, min_max):
        if isinstance(fix, bool) and fix:
            min_vals = min(vals)
            max_vals = max(vals)
            mmin = min(min_max[0], min_vals - plot_margin * abs(min_vals))
            mmax = max(min_max[1], max_vals + plot_margin * abs(max_vals))
            return (mmin, mmax)
        elif not isinstance(fix, bool) and len(fix) == 2:
            return fix
        else:
            return min_max

    # Compute PDPs for all variables
    pred_dict = {}
    for v in incl:
        result = compute_pdp_single(v)
        pred_dict[v] = result
        min_max = calc_ylim(result["prediction"].to_list(), min_max)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        result = compute_pdp_interaction(vl[0], vl[1])
        pred_dict[v] = result
        if sum(is_num) < 2:
            min_max = calc_ylim(result["prediction"].to_list(), min_max)

    # Create plots
    plot_list = []
    for v in incl:
        plot_data = pred_dict[v]
        is_num = _is_numeric_with_many_unique(data, v, 5)

        if is_num:
            p = ggplot(plot_data, aes(x=v, y="prediction")) + geom_line(color="steelblue")
        else:
            p = (
                ggplot(plot_data, aes(x=v, y="prediction"))
                + geom_line(color="steelblue", group=1)
                + geom_point(color="steelblue", size=3)
            )

        p = p + labs(x="", y="Prediction") + ggtitle(v) + theme_bw()

        if isinstance(min_max, tuple) and min_max[0] != float("inf"):
            p = p + scale_y_continuous(limits=min_max)

        if hline_val is not None:
            p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

        plot_list.append(p)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        plot_data = pred_dict[v]

        if sum(is_num) == 2:
            # Two numeric: line plot with color (sliced)
            plot_data = plot_data.with_columns(pl.col(vl[1]).round(2).cast(pl.Utf8).alias(vl[1]))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line()
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
        elif sum(is_num) == 1:
            # One numeric, one categorical
            if is_num[1]:
                vl = [vl[1], vl[0]]
            plot_data = plot_data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line()
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
            if isinstance(min_max, tuple) and min_max[0] != float("inf"):
                p = p + scale_y_continuous(limits=min_max)
            if hline_val is not None:
                p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")
        else:
            # Both categorical
            plot_data = plot_data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line(aes(group=vl[1]))
                + geom_point(size=3)
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
            if isinstance(min_max, tuple) and min_max[0] != float("inf"):
                p = p + scale_y_continuous(limits=min_max)
            if hline_val is not None:
                p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

        plot_list.append(p)

    runtime = time.time() - start_time
    plot = _compose_plots(plot_list, ncol=ncol)

    # Add runtime caption if we have plots
    if plot is not None:
        plot = plot + labs(caption=f"Runtime: {runtime:.2f}s | Mode: {mode}")

    return plot, pred_dict, runtime


def pdp_sm(
    fitted,
    data: pl.DataFrame | pd.DataFrame,
    incl=None,
    excl=[],
    incl_int=[],
    mode="pdp",
    n_sample=2000,
    grid_resolution=50,
    minq=0.05,
    maxq=0.95,
    interaction_slices=5,
    fix=True,
    hline=True,
    ncol=2,
):
    """
    Generate Partial Dependence Plots (PDP) for statsmodels regression models.

    Parameters
    ----------
    fitted : A fitted statsmodels model (OLS or Logistic)
    data : Polars or Pandas DataFrame with data used for estimation
    incl : A list of column names to generate PDP plots for
    excl : A list of column names to exclude from plotting
    incl_int : A list of ":" separated column names for interaction plots
    mode : str, "fast" or "pdp"
        "fast" - uses sim_prediction (mean/mode for other vars)
        "pdp" - true PDP: samples rows, replaces feature values, averages predictions
    n_sample : int
        Number of samples to use for PDP mode
    grid_resolution : int
        Number of grid points for numeric variables
    minq : float
        Quantile for minimum value of numeric variables
    maxq : float
        Quantile for maximum value of numeric variables
    interaction_slices : int
        Number of slices for numeric-numeric interactions
    fix : Logical or tuple for y-axis limits
    hline : Logical or float for horizontal line
    ncol : int
        Number of columns in plot grid

    Returns
    -------
    tuple: (plot, data_dict, runtime_seconds)
        - plot: plotnine plot composition
        - data_dict: dict of DataFrames with underlying PDP data
        - runtime_seconds: total computation time
    """
    start_time = time.time()

    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        model = fitted

    # Convert to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    rvar = model.endog_names

    # Get explanatory variables
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
        return None, {}, 0.0

    # Get all evars for prediction
    all_evars = extract_evars(model, data.columns)

    # Get base data with all evars
    base_data = data.select(all_evars).drop_nulls()

    # Sample for PDP mode
    n_obs = base_data.height
    if mode == "pdp":
        sample_size = min(n_sample, n_obs)
        if sample_size < n_obs:
            sample_data = base_data.sample(sample_size, seed=1234)
        else:
            sample_data = base_data
    else:
        sample_data = base_data

    # Helper to build grid
    def build_grid(var):
        col = data[var]
        if _is_categorical(data, var, 5):
            return col.unique().drop_nulls().to_list()
        else:
            nu = col.n_unique()
            min_val = col.quantile(minq)
            max_val = col.quantile(maxq)
            return np.linspace(min_val, max_val, min(nu, grid_resolution)).tolist()

    # Compute PDP for single variable
    def compute_pdp_single(var):
        grid_vals = build_grid(var)
        predictions = []

        if mode == "pdp":
            # Get all evars from model for constructing prediction data
            evars = extract_evars(model, data.columns)

            for gv in grid_vals:
                # Replace var in sample_data while keeping all other vars
                modified = sample_data.select(evars).with_columns(
                    pl.lit(gv).cast(sample_data[var].dtype).alias(var)
                )
                modified_pd = to_pandas_with_categories(modified, data)
                preds = fitted.predict(modified_pd)
                predictions.append(np.mean(preds))
            return pl.DataFrame({var: grid_vals, "prediction": predictions})
        else:
            # Fast mode
            iplot = sim_prediction(data, vary=var, nnv=grid_resolution, minq=minq, maxq=maxq)
            iplot_pd = to_pandas_with_categories(iplot, data)
            preds = fitted.predict(iplot_pd)
            return iplot.with_columns(prediction=pl.Series(list(preds)))

    # Compute PDP for interaction
    def compute_pdp_interaction(var1, var2):
        is_num1 = _is_numeric_with_many_unique(data, var1, 5)
        is_num2 = _is_numeric_with_many_unique(data, var2, 5)

        grid1 = build_grid(var1)
        grid2 = build_grid(var2)

        if is_num1 and is_num2:
            grid2 = np.linspace(
                data[var2].quantile(minq),
                data[var2].quantile(maxq),
                interaction_slices,
            ).tolist()

        schema = {var1: data[var1].dtype, var2: data[var2].dtype}
        grid_df = expand_grid({var1: grid1, var2: grid2}, schema)
        predictions = []

        # Get all evars from model
        evars = extract_evars(model, data.columns)

        if mode == "pdp":
            for row in grid_df.iter_rows(named=True):
                modified = sample_data.select(evars).with_columns(
                    pl.lit(row[var1]).cast(sample_data[var1].dtype).alias(var1),
                    pl.lit(row[var2]).cast(sample_data[var2].dtype).alias(var2),
                )
                modified_pd = to_pandas_with_categories(modified, data)
                preds = fitted.predict(modified_pd)
                predictions.append(np.mean(preds))
        else:
            for row in grid_df.iter_rows(named=True):
                iplot = sim_prediction(data, vary={var1: [row[var1]], var2: [row[var2]]})
                iplot_pd = to_pandas_with_categories(iplot, data)
                preds = fitted.predict(iplot_pd)
                predictions.append(preds[0])

        return grid_df.with_columns(prediction=pl.Series(predictions))

    # Plot margin and y-axis limits
    plot_margin = 0.025

    if isinstance(hline, bool):
        if hline and rvar in data.columns:
            col = data[rvar]
            # For numeric columns, use mean; for categorical, compute proportion
            if col.dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                hline_val = col.mean()
            else:
                # Categorical: compute proportion of positive class
                # Use type-appropriate values for comparison
                if col.dtype == pl.Boolean:
                    hline_val = col.mean()
                else:
                    # String/categorical: check for common positive labels
                    positive_str = ["Yes", "yes", "YES", "1", "True", "true"]
                    hline_val = col.cast(pl.Utf8).is_in(positive_str).mean()

            if hline_val is not None:
                min_max = (
                    hline_val - plot_margin * abs(hline_val),
                    hline_val + plot_margin * abs(hline_val),
                )
            else:
                min_max = (float("inf"), float("-inf"))
        else:
            hline_val = None
            min_max = (float("inf"), float("-inf"))
    else:
        hline_val = hline
        min_max = (hline - plot_margin * abs(hline), hline + plot_margin * abs(hline))

    def calc_ylim(vals, min_max):
        if isinstance(fix, bool) and fix:
            min_vals = min(vals)
            max_vals = max(vals)
            mmin = min(min_max[0], min_vals - plot_margin * abs(min_vals))
            mmax = max(min_max[1], max_vals + plot_margin * abs(max_vals))
            return (mmin, mmax)
        elif not isinstance(fix, bool) and len(fix) == 2:
            return fix
        else:
            return min_max

    # Compute PDPs
    pred_dict = {}
    for v in incl:
        result = compute_pdp_single(v)
        pred_dict[v] = result
        min_max = calc_ylim(result["prediction"].to_list(), min_max)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        result = compute_pdp_interaction(vl[0], vl[1])
        pred_dict[v] = result
        if sum(is_num) < 2:
            min_max = calc_ylim(result["prediction"].to_list(), min_max)

    # Create plots
    plot_list = []
    for v in incl:
        plot_data = pred_dict[v]
        is_num = _is_numeric_with_many_unique(data, v, 5)

        if is_num:
            p = ggplot(plot_data, aes(x=v, y="prediction")) + geom_line(color="steelblue")
        else:
            p = (
                ggplot(plot_data, aes(x=v, y="prediction"))
                + geom_line(color="steelblue", group=1)
                + geom_point(color="steelblue", size=3)
            )

        p = p + labs(x="", y="Prediction") + ggtitle(v) + theme_bw()

        if isinstance(min_max, tuple) and min_max[0] != float("inf"):
            p = p + scale_y_continuous(limits=min_max)

        if hline_val is not None:
            p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

        plot_list.append(p)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        plot_data = pred_dict[v]

        if sum(is_num) == 2:
            plot_data = plot_data.with_columns(pl.col(vl[1]).round(2).cast(pl.Utf8).alias(vl[1]))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line()
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
        elif sum(is_num) == 1:
            if is_num[1]:
                vl = [vl[1], vl[0]]
            plot_data = plot_data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line()
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
            if isinstance(min_max, tuple) and min_max[0] != float("inf"):
                p = p + scale_y_continuous(limits=min_max)
            if hline_val is not None:
                p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")
        else:
            plot_data = plot_data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
            p = (
                ggplot(plot_data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line(aes(group=vl[1]))
                + geom_point(size=3)
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(v)
                + theme_bw()
            )
            if isinstance(min_max, tuple) and min_max[0] != float("inf"):
                p = p + scale_y_continuous(limits=min_max)
            if hline_val is not None:
                p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

        plot_list.append(p)

    runtime = time.time() - start_time
    plot = _compose_plots(plot_list, ncol=ncol)

    if plot is not None:
        plot = plot + labs(caption=f"Runtime: {runtime:.2f}s | Mode: {mode}")

    return plot, pred_dict, runtime


def vimp_plot_sm(fitted, data: pl.DataFrame | pd.DataFrame, rep=10, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    statsmodels library.

    Parameters
    ----------
    fitted : A fitted statsmodels object
    data : Polars DataFrame with data used for estimation
    rep: int
        The number of times to resample and calculate the permutation importance
    ret: bool
        Return the variable importance table as a sorted DataFrame

    Returns
    -------
    plotnine plot or polars DataFrame if ret=True
    """
    fw = None
    if hasattr(fitted, "model"):
        model = fitted.model
        if hasattr(model, "_has_freq_weights") and model._has_freq_weights:
            fw = model.freq_weights
    else:
        return "This function requires a fitted linear or logistic regression"

    # Convert pandas to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    rvar = extract_rvar(model, data.columns)
    evars = extract_evars(model, data.columns)
    data = data.select([rvar] + evars).drop_nulls()

    if len(model.endog) != data.height:
        raise Exception(
            "The number of rows in the DataFrame should be the same as the number of rows in the data used to estimate the model"
        )

    def imp_calc_reg(base, pred):
        return base - _calc_r_squared(model.endog, pred)

    def imp_calc_logit(base, pred):
        return base - auc(model.endog, pred, weights=fw)

    # Calculate the baseline performance
    data_pd = data.select(evars).to_pandas()
    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        baseline_fit = auc(model.endog, fitted.predict(data_pd), weights=fw)
        imp_calc = imp_calc_logit
        xlab = "Importance (AUC decrease)"
    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        baseline_fit = _calc_r_squared(model.endog, fitted.predict(data_pd))
        imp_calc = imp_calc_reg
        xlab = "Importance (R-square decrease)"
    else:
        return "This model type is not supported. For sklearn models use vimp_plot_sk"

    # Initialize importance values
    importance_values = {v: 0.0 for v in evars}

    # Iterate over each feature
    for i in range(rep):
        for feature in evars:
            # Shuffle the feature column
            shuffled = data.with_columns(pl.col(feature).shuffle(seed=i).alias(feature))
            shuffled_pd = shuffled.select(evars).to_pandas()
            importance_values[feature] += imp_calc(baseline_fit, fitted.predict(shuffled_pd))

    # Average importance
    importance_values = {k: v / rep for k, v in importance_values.items()}

    # Create DataFrame for plotting
    imp_df = pl.DataFrame(
        {
            "variable": list(importance_values.keys()),
            "importance": list(importance_values.values()),
        }
    ).sort("importance")

    # Create horizontal bar plot
    p = (
        ggplot(imp_df, aes(x="variable", y="importance"))
        + geom_bar(stat="identity", fill="slateblue")
        + coord_flip()
        + scale_x_discrete(limits=imp_df["variable"].to_list())
        + labs(x="", y=xlab)
        + ggtitle("Permutation Importance")
        + theme_bw()
        + theme(figure_size=(6, max(5, len(imp_df) * 0.4)))
    )

    if ret:
        return (imp_df.sort("importance", descending=True), p)

    return (None, p)


def vimp_plot_sk(model, rep=5, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    sklearn library. Handles categorical variables by shuffling the original
    variable rather than dummy-encoded versions.

    Parameters
    ----------
    model : A pyrsm model object with fitted sklearn model
    rep: int
        The number of times to resample and calculate permutation importance
    ret: bool
        Return the variable importance table as a sorted DataFrame

    Returns
    -------
    plotnine plot or polars DataFrame if ret=True
    """
    rvar = model.rvar
    evars = model.evar

    # Get data as polars
    data = model.data.select([rvar] + evars).drop_nulls()

    def imp_calc_reg(base, pred):
        return base - _calc_r_squared(data[rvar], pred)

    def imp_calc_clf(base, pred):
        return base - auc(data[rvar].to_list(), pred)

    # Calculate the baseline performance
    if hasattr(model.fitted, "classes_"):
        xlab = "Importance (AUC decrease)"
        baseline_pred = model.predict(data.select(evars))
        baseline_fit = auc(data[rvar].to_list(), baseline_pred["prediction"].to_list())
        imp_calc = imp_calc_clf
    else:
        baseline_pred = model.predict(data.select(evars))
        baseline_fit = _calc_r_squared(data[rvar], baseline_pred["prediction"])
        imp_calc = imp_calc_reg
        xlab = "Importance (R-square decrease)"

    # Initialize importance values
    importance_values = {v: 0.0 for v in evars}

    # Iterate over each feature
    for i in range(rep):
        for feature in evars:
            # Shuffle the feature column
            permuted = data.with_columns(pl.col(feature).shuffle(seed=i).alias(feature))
            pred_result = model.predict(permuted.select(evars))
            importance_values[feature] += imp_calc(
                baseline_fit, pred_result["prediction"].to_list()
            )

    # Average importance
    importance_values = {k: v / rep for k, v in importance_values.items()}

    # Create DataFrame for plotting
    imp_df = pl.DataFrame(
        {
            "variable": list(importance_values.keys()),
            "importance": list(importance_values.values()),
        }
    ).sort("importance")

    # Create horizontal bar plot
    p = (
        ggplot(imp_df, aes(x="variable", y="importance"))
        + geom_bar(stat="identity", fill="slateblue")
        + coord_flip()
        + scale_x_discrete(limits=imp_df["variable"].to_list())
        + labs(x="", y=xlab)
        + ggtitle("Permutation Importance")
        + theme_bw()
        + theme(figure_size=(6, max(5, len(imp_df) * 0.4)))
    )

    return (p, imp_df.sort("importance", descending=True))


def vimp_plot_sklearn(fitted, X: pl.DataFrame, y, rep=5, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    sklearn library using sklearn's built-in permutation_importance.

    Parameters
    ----------
    fitted : A fitted sklearn object
    X : Polars DataFrame with explanatory variables (features)
    y : Series or list with response variable (target)
    rep: int
        The number of times to resample and calculate permutation importance
    ret: bool
        Return the variable importance table as a sorted DataFrame

    Returns
    -------
    plotnine plot or polars DataFrame if ret=True
    """
    if hasattr(fitted, "classes_"):
        scoring = "roc_auc"
        xlab = "Importance (AUC decrease)"
    else:
        scoring = "r2"
        xlab = "Importance (R-square decrease)"

    # sklearn needs pandas/numpy
    if isinstance(X, pd.DataFrame):
        X_pd = X
    else:
        X_pd = X.to_pandas()
    y_list = y.to_list() if isinstance(y, pl.Series) else list(y)

    imp = permutation_importance(
        fitted, X_pd, y_list, scoring=scoring, n_repeats=rep, random_state=1234
    )

    # Create importance DataFrame
    imp_df = pl.DataFrame(
        {
            "variable": list(fitted.feature_names_in_),
            "importance": imp.importances_mean.tolist(),
        }
    ).sort("importance")

    # Create horizontal bar plot
    p = (
        ggplot(imp_df, aes(x="variable", y="importance"))
        + geom_bar(stat="identity", fill="slateblue")
        + coord_flip()
        + scale_x_discrete(limits=imp_df["variable"].to_list())
        + labs(x="", y=xlab)
        + ggtitle("Permutation Importance")
        + theme_bw()
        + theme(figure_size=(6, max(5, len(imp_df) * 0.4)))
    )

    if ret:
        return (p, imp_df.sort("importance", descending=True))
    else:
        return (p, None)
