import numpy as np
import polars as pl
from plotnine import (
    aes,
    geom_bar,
    geom_hline,
    geom_line,
    geom_point,
    geom_segment,
    geom_vline,
    ggplot,
    ggtitle,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)
from sklearn import metrics

# from pyrsm import xtile, bincode
from ..bins import xtile, bincode
from ..utils import ifelse, table2data, check_dataframe


def calc_qnt(df, rvar, lev, pred, qnt=10):
    """
    Create quantiles and calculate input to use for lift and gains charts

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create

    Returns
    -------
    Polars dataframe
        Response metrics per quantile. Used as input for lift and gains charts
    """
    df = check_dataframe(df).select([rvar, pred])
    bins = xtile(df[pred].to_numpy(), qnt)
    rvar_int = pl.when(df[rvar] == lev).then(1).when(df[rvar].is_null()).then(None).otherwise(0)

    df = df.with_columns([pl.Series("bins", bins), rvar_int.alias("rvar_int")])

    perf_df = (
        df.group_by("bins")
        .agg([pl.len().alias("nr_obs"), pl.col("rvar_int").sum().alias("nr_resp")])
        .sort("bins")
    )

    # flip if needed
    if perf_df["nr_resp"][1] < perf_df["nr_resp"][-1]:
        perf_df = perf_df.sort("bins", descending=True)

    total_obs = perf_df["nr_obs"].sum()
    perf_df = perf_df.with_columns(
        [
            pl.col("nr_obs").cum_sum().alias("cum_obs"),
            pl.col("nr_resp").cum_sum().alias("cum_resp"),
        ]
    ).with_columns((pl.col("cum_obs") / total_obs).alias("cum_prop"))

    return perf_df


def gains_tab(df, rvar, lev, pred, qnt=10):
    """
    Calculate cumulative gains using the cum_resp column

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create

    Returns
    -------
    Polars dataframe
        Gains measures per quantile. Input for gains chart
    """
    df = calc_qnt(df, rvar, lev, pred, qnt=qnt)
    total_resp = df["cum_resp"][-1]
    df = df.with_columns((pl.col("cum_resp") / total_resp).alias("cum_gains"))
    df = df.select(["cum_prop", "cum_gains"])
    df0 = pl.DataFrame({"cum_prop": [0.0], "cum_gains": [0.0]})
    return pl.concat([df0, df])


def lift_tab(df, rvar, lev, pred, qnt=10):
    """
    Calculate cumulative lift using the cum_resp and the cum_obs column

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create

    Returns
    -------
    Polars dataframe
        Lift measures per quantile. Input for lift chart
    """
    df = calc_qnt(df, rvar, lev, pred, qnt=qnt)
    df = df.with_columns((pl.col("cum_resp") / pl.col("cum_obs")).alias("cum_resp_rate"))
    final_rate = df["cum_resp_rate"][-1]
    df = df.with_columns((pl.col("cum_resp_rate") / final_rate).alias("cum_lift"))
    return df.select(["cum_prop", "cum_lift"])


def confusion(df, rvar, lev, pred, cost=1, margin=2):
    """
    Calculate TP, FP, TN, FN, and contact

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    TP : int
        Number of True Positive predictions
    FP : int
        Number of False Positive predictions
    TN : int
        Number of True Negative predictions
    FN : int
        Number of False Negative predictions
    contact: float
        Proportion of cases to act on based on the cost/margin ratio
    """
    if isinstance(pred, list | tuple) and len(pred) > 1:
        return "This function can only take one predictor variables at time"

    df = check_dataframe(df)
    break_even = cost / margin
    gtbe = df[pred].to_numpy() > break_even
    pos = df[rvar].to_numpy() == lev
    TP = np.where(gtbe & pos, 1, 0).sum()
    FP = np.where(gtbe & ~pos, 1, 0).sum()
    TN = np.where(~gtbe & ~pos, 1, 0).sum()
    FN = np.where(~gtbe & pos, 1, 0).sum()
    contact = (TP + FP) / (TP + FP + TN + FN)
    return TP, FP, TN, FN, contact


def uplift_tab(df, rvar, lev, pred, tvar, tlev, scale=1, qnt=10):
    """
    Calculate an Uplift table

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    tvar : str
        Name of the treatment variable column in df
    tlev : str
        Name of the 'success' level in tvar
    scale : float
        Scaling factor to use in calculations
    qnt : int
        Number of quantiles to create

    Returns
    -------
    Polars dataframe
        Incremental uplift per quantile. Input for uplift charts
    """

    def local_xtile(x, treatment, n=qnt, rev=True):
        x = np.array(x)
        treatment = np.array(treatment)
        breaks = np.concatenate(
            (
                np.array([-np.inf]),
                np.quantile(x[treatment], np.arange(0, n + 1) / n, method="linear")[1:-1],
                np.array([np.inf]),
            )
        )

        if len(np.unique(breaks)) == len(breaks):
            # Use polars cut
            s = pl.Series("x", x)
            cut_result = s.cut(breaks[1:-1], labels=[str(i) for i in range(1, n + 1)])
            bins = cut_result.cast(pl.Int64).to_numpy()
        else:
            bins = bincode(x, breaks)

        if rev is True:
            bins = (n + 1) - bins

        return bins

    df = check_dataframe(df)
    rvar_bool = df[rvar] == lev
    tvar_bool = df[tvar] == tlev

    bins = local_xtile(df[pred].to_numpy(), tvar_bool.to_numpy(), n=qnt, rev=True)

    df = df.with_columns(
        [
            rvar_bool.alias("rvar_bool"),
            tvar_bool.alias("tvar_bool"),
            pl.Series("bins", bins),
            (tvar_bool & rvar_bool).alias("T_resp"),
            (~tvar_bool & rvar_bool).alias("C_resp"),
            (~tvar_bool).alias("C_n"),
        ]
    )

    # Group by bins and aggregate
    tab = (
        df.group_by("bins")
        .agg(
            [
                pl.len().alias("nr_obs"),
                pl.col("rvar_bool").sum().alias("nr_resp"),
                pl.col("T_resp").sum().alias("T_resp"),
                pl.col("tvar_bool").sum().alias("T_n"),
                pl.col("C_resp").sum().alias("C_resp"),
                pl.col("C_n").sum().alias("C_n"),
            ]
        )
        .sort("bins")
    )

    # Calculate uplift per bin (before cumsum)
    tab = tab.with_columns(
        (pl.col("T_resp") / pl.col("T_n") - pl.col("C_resp") / pl.col("C_n")).alias("uplift")
    )

    # Calculate cumulative values and proportions
    tab = tab.with_columns(
        [
            (pl.lit(1).cum_sum() / qnt).alias("cum_prop"),
            (pl.col("T_resp").cum_sum() * scale).alias("T_resp"),
            (pl.col("T_n").cum_sum() * scale).alias("T_n"),
            (pl.col("C_resp").cum_sum() * scale).alias("C_resp"),
            (pl.col("C_n").cum_sum() * scale).alias("C_n"),
        ]
    )

    # Calculate incremental response and uplift
    tab = tab.with_columns(
        (pl.col("T_resp") - pl.col("C_resp") * pl.col("T_n") / pl.col("C_n")).alias(
            "incremental_resp"
        )
    )

    T_n_max = tab["T_n"].max()
    tab = tab.with_columns(
        [
            (pl.col("incremental_resp") / T_n_max * 100).alias("inc_uplift"),
            pl.lit(pred).alias("pred"),
        ]
    )

    return tab.select(
        [
            "pred",
            "bins",
            "cum_prop",
            "T_resp",
            "T_n",
            "C_resp",
            "C_n",
            "incremental_resp",
            "inc_uplift",
            "uplift",
        ]
    )


def inc_uplift_plot(df, rvar, lev, pred, tvar, tlev, scale=1, qnt=10):
    """
    Plot an Incremental Uplift chart

    Parameters
    ----------
    df : Polars dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    tvar : str
        Name of the treatment variable column in df
    tlev : str
        Name of the 'success' level in tvar
    scale : float
        Scaling factor to use in calculations
    qnt : int
        Number of quantiles to create

    Returns
    -------
    plotnine ggplot object
        Plot of Incremental Uplift per quantile
    """
    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)
    has_groups = len(pred) > 1 or len(dct.keys()) > 1

    rd = [
        pl.concat(
            [
                pl.DataFrame({"cum_prop": [0.0], "inc_uplift": [0.0], "pred": [p]}),
                uplift_tab(dct[k], rvar, lev, p, tvar, tlev, scale=scale, qnt=qnt)
                .with_columns(pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("pred"))
                .select(["cum_prop", "inc_uplift", "pred"]),
            ]
        )
        for k in dct.keys()
        for p in pred
    ]

    plot_df = pl.concat(rd)
    yend = rd[0]["inc_uplift"][-1]

    # Create baseline data for diagonal line
    baseline = pl.DataFrame({"cum_prop": [0.0, 1.0], "inc_uplift": [0.0, yend]})

    if has_groups:
        p = (
            ggplot(plot_df, aes(x="cum_prop", y="inc_uplift", color="pred"))
            + geom_line()
            + geom_point()
        )
    else:
        p = ggplot(plot_df, aes(x="cum_prop", y="inc_uplift")) + geom_line() + geom_point()

    p = (
        p
        + geom_line(
            data=baseline,
            mapping=aes(x="cum_prop", y="inc_uplift"),
            linetype="dashed",
            color="steelblue",
            inherit_aes=False,
        )
        + scale_x_continuous(labels=lambda x: [f"{v:.0%}" for v in x])
        + scale_y_continuous(labels=lambda y: [f"{v:.0%}" for v in [yi / 100 for yi in y]])
        + labs(x="Percentage of population targeted", y="Incremental Uplift", color="")
        + theme_bw()
    )

    return p


def inc_profit_tab(
    df,
    rvar,
    lev,
    pred,
    tvar,
    tlev,
    cost=1,
    margin=2,
    scale=1,
    qnt=10,
):
    """
    Tabulate Incremental Profit for Uplift modeling

    Parameters
    ----------
    df : Pandas dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    tvar : str
        Name of the treatment variable column in df
    tlev : str
        Name of the 'success' level in tvar
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    scale : float
        Scaling factor to use in calculations
    qnt : int
        Number of quantiles to create

    Returns
    -------
    Polars Dataframe
        Incremental Uplift per quantile
    """

    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)

    rd = [
        pl.concat(
            [
                pl.DataFrame(
                    {"cum_prop": [0.0], "incremental_resp": [0.0], "T_n": [0.0], "pred": [p]}
                ),
                uplift_tab(dct[k], rvar, lev, p, tvar, tlev, scale=scale, qnt=qnt)
                .with_columns(pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("predictor"))
                .select(["cum_prop", "incremental_resp", "T_n", "pred"]),
            ]
        )
        for k in dct.keys()
        for p in pred
    ]

    rd = pl.concat(rd)
    rd = rd.with_columns(
        (pl.col("incremental_resp") * margin - pl.col("T_n") * cost).alias("inc_profit")
    )
    return rd


def inc_profit_plot(
    df,
    rvar,
    lev,
    pred,
    tvar,
    tlev,
    cost=1,
    margin=2,
    scale=1,
    qnt=10,
    contact=True,
    prn=False,
):
    """
    Plot an Incremental Profit chart for Uplift modeling

    Parameters
    ----------
    df : Polars dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    tvar : str
        Name of the treatment variable column in df
    tlev : str
        Name of the 'success' level in tvar
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    scale : float
        Scaling factor to use in calculations
    qnt : int
        Number of quantiles to create
    contact : bool
        Plot a vertical line that shows the optimal contact level.
    prn : bool
        Print the maximum incremental profit value(s) (default is True, requires contact=True)

    Returns
    -------
    plotnine ggplot object
        Plot of Incremental Profit per quantile
    """
    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)
    has_groups = len(pred) > 1 or len(dct.keys()) > 1

    plot_df = [
        inc_profit_tab(dct[k], rvar, lev, p, tvar, tlev, cost, margin, scale, qnt).with_columns(
            pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("predictor")
        )
        for k in dct.keys()
        for p in pred
    ]
    plot_df = pl.concat(plot_df).drop("pred")

    if has_groups:
        p = (
            ggplot(plot_df, aes(x="cum_prop", y="inc_profit", color="predictor"))
            + geom_line()
            + geom_point()
        )
    else:
        p = ggplot(plot_df, aes(x="cum_prop", y="inc_profit")) + geom_line() + geom_point()

    p = (
        p
        + geom_hline(yintercept=0, linetype="dashed")
        + scale_x_continuous(labels=lambda x: [f"{v:.0%}" for v in x])
        + scale_y_continuous(labels=lambda y: [f"{int(v):,}" for v in y])
        + labs(x="Percentage of population targeted", y="Incremental Profit", color="")
        + theme_bw()
    )

    if contact:
        cnf = [
            confusion(
                dct[k].filter(pl.col(tvar) == tlev),
                rvar,
                lev,
                pr,
                cost=cost,
                margin=margin,
            )[-1]
            for k in dct.keys()
            for pr in pred
        ]
        if prn:
            print(plot_df)
        # Add vertical lines for contact levels < 1
        for cnf_val in filter(lambda x: x < 1, cnf):
            p = p + geom_vline(xintercept=cnf_val, linetype="dashed")

    return p


def uplift_plot(df, rvar, lev, pred, tvar, tlev, qnt=10):
    """
    Plot an Uplift bar chart

    Parameters
    ----------
    df : Polars dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    tvar : str
        Name of the treatment variable column in df
    tlev : str
        Name of the 'success' level in tvar
    qnt : int
        Number of quantiles to create

    Returns
    -------
    plotnine ggplot object
        Plot of Uplift per quantile
    """
    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)
    has_groups = len(pred) > 1 or len(dct.keys()) > 1

    rd = [
        uplift_tab(dct[k], rvar, lev, p, tvar, tlev, qnt=qnt).with_columns(
            pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("predictor")
        )
        for k in dct.keys()
        for p in pred
    ]

    rd = pl.concat(rd)
    # Convert cum_prop to string for categorical x-axis
    rd = rd.with_columns(pl.col("cum_prop").round(2).cast(pl.Utf8).alias("cum_prop_str"))

    if has_groups:
        p = ggplot(rd, aes(x="cum_prop_str", y="uplift", fill="predictor")) + geom_bar(
            stat="identity", position="dodge"
        )
    else:
        p = ggplot(rd, aes(x="cum_prop_str", y="uplift")) + geom_bar(
            stat="identity", fill="steelblue"
        )

    p = (
        p
        + scale_y_continuous(labels=lambda y: [f"{v:.0%}" for v in y])
        + labs(x="Percentage of population targeted", y="Uplift", fill="")
        + theme_bw()
    )

    return p


def profit_max(df, rvar, lev, pred, cost=1, margin=2, scale=1):
    """
    Calculate the maximum profit using a dataframe as input

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    scale : float
        Scaling factor to use in calculations

    Returns
    -------
    float
        Measure of optimal performance (e.g., profit) based on the specified cost and margin information
    """
    TP, FP, TN, FN, contact = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    return scale * (margin * TP - cost * (TP + FP))


def profit(rvar, pred, lev=1, cost=1, margin=2, scale=1):
    """
    Calculate the maximum profit using series as input. Provides the same results as profit_max

    Parameters
    ----------
    rvar : Pandas series, Polars series, or numpy array
        Column with the response variable
    pred : Pandas series, Polars series, or numpy array
        Column with model predictions
    lev : str
        Name of the 'success' level in rvar
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    scale : float
        Scaling factor to use in calculations

    Returns
    -------
    float
        Measure of optimal performace (e.g., profit) based on the specified cost and margin information
    """
    if isinstance(rvar, pl.Series):
        rvar = rvar.to_numpy()
    if isinstance(pred, pl.Series):
        pred = pred.to_numpy()
    rvar = np.array(rvar)
    pred = np.array(pred)

    break_even = cost / margin
    TP = ((rvar == lev) & (pred > break_even)).sum()
    FP = ((rvar != lev) & (pred > break_even)).sum()
    return scale * (margin * TP - cost * (TP + FP))


def ROME_max(df, rvar, lev, pred, cost=1, margin=2):
    """
    Calculate the maximum Return on Marketing Expenditures

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    float
        Measure of optimal performace based on the specified cost and margin information
    """
    TP, FP, TN, FN, contact = confusion(df, rvar, lev, pred, cost=cost, margin=margin)
    profit = margin * TP - cost * (TP + FP)
    return profit / (cost * (TP + FP))


def ROME(pred, rvar, lev, cost=1, margin=2):
    """
    Calculate the maximum Return on Marketing Expenditures using series as input.
    Provides the same results as ROME_max

    Parameters
    ----------
    pred : Pandas series, Polars series, or numpy array
        Column with model predictions
    rvar : Pandas series, Polars series, or numpy array
        Column with the response variable
    lev : str
        Name of the 'success' level in rvar
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    float
        Measure of optimal performace (e.g., profit) based on the specified cost and margin information
    """
    if isinstance(rvar, pl.Series):
        rvar = rvar.to_numpy()
    if isinstance(pred, pl.Series):
        pred = pred.to_numpy()
    rvar = np.array(rvar)
    pred = np.array(pred)

    break_even = cost / margin
    TP = ((rvar == lev) & (pred > break_even)).sum()
    FP = ((rvar != lev) & (pred > break_even)).sum()
    profit = margin * TP - cost * (TP + FP)
    return profit / (cost * (TP + FP))


def profit_tab(df, rvar, lev, pred, qnt=10, cost=1, margin=2, scale=1):
    """
    Calculate table with profit per quantile

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    scale : float
        Scaling factor to use in calculations

    Returns
    -------
    Polars dataframe
        Profit per quantile. Input for profit chart
    """
    df = calc_qnt(df, rvar, lev, pred, qnt=qnt)
    df = df.with_columns(
        ((margin * pl.col("cum_resp") - cost * pl.col("cum_obs")) * scale)
        .cast(pl.Float64)
        .alias("cum_profit")
    )
    df = df.select(["cum_prop", "cum_profit"])
    df0 = pl.DataFrame({"cum_prop": [0.0], "cum_profit": [0.0]})
    return pl.concat([df0, df])


def ROME_tab(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """
    Calculate table with Return on Marketing Expenditures per quantile

    Parameters
    ----------
    df : Pandas or Polars dataframe
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    Polars dataframe
        ROME quantile. Input for ROME chart
    """
    df = calc_qnt(df, rvar, lev, pred, qnt=qnt)
    df = df.with_columns(
        (margin * pl.col("cum_resp") - cost * pl.col("cum_obs")).alias("cum_profit"),
        (cost * pl.col("cum_obs")).alias("cum_cost"),
    )
    df = df.with_columns(
        ((margin * pl.col("cum_resp") - pl.col("cum_cost")) / pl.col("cum_cost")).alias("ROME")
    )
    return df.select(["cum_prop", "ROME"])


def profit_plot(
    df,
    rvar,
    lev,
    pred,
    qnt=10,
    cost=1,
    margin=2,
    scale=1,
    contact=True,
    prn=True,
):
    """
    Plot a profit curve

    Parameters
    ----------
    df : Polars dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    scale : float
        Scaling factor to use in calculations
    contact : bool
        Plot a vertical line that shows the optimal contact level. Requires
        that `pred` is a series of probabilities. Values equal to 1 (100% contact)
        will not be plotted
    prn : bool
        Print the maximum profit value (default is True, requires contact=True)

    Returns
    -------
    plotnine ggplot object
        Plot of profits per quantile

    Examples
    --------
    profit_plot(df, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    profit_plot(df, "buyer", "yes", ["pred_a", "pred_b"], cost=0.5, margin=6)
    dct = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    profit_plot(dct, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    """
    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)
    has_groups = len(pred) > 1 or len(dct.keys()) > 1

    plot_df = [
        profit_tab(
            dct[k], rvar, lev, p, qnt=qnt, cost=cost, margin=margin, scale=scale
        ).with_columns(pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("predictor"))
        for k in dct.keys()
        for p in pred
    ]
    plot_df = pl.concat(plot_df)

    if has_groups:
        p = (
            ggplot(plot_df, aes(x="cum_prop", y="cum_profit", color="predictor"))
            + geom_line()
            + geom_point()
        )
    else:
        p = ggplot(plot_df, aes(x="cum_prop", y="cum_profit")) + geom_line() + geom_point()

    p = (
        p
        + geom_hline(yintercept=0, linetype="dashed")
        + scale_x_continuous(labels=lambda x: [f"{v:.0%}" for v in x])
        + scale_y_continuous(labels=lambda y: [f"{int(v):,}" for v in y])
        + labs(x="Percentage of population targeted", y="Profit", color="")
        + theme_bw()
    )

    if contact:
        cnf = [
            confusion(dct[k], rvar, lev, pr, cost=cost, margin=margin)[-1]
            for k in dct.keys()
            for pr in pred
        ]
        prof = [
            profit_max(dct[k], rvar, lev, pr, cost=cost, margin=margin, scale=scale)
            for k in dct.keys()
            for pr in pred
        ]
        if prn:
            print(prof)
        # Add vertical and horizontal lines for contact levels < 1
        for i, cnf_val in enumerate(filter(lambda x: x < 1, cnf)):
            p = p + geom_vline(xintercept=cnf_val, linetype="dashed")
            p = p + geom_hline(yintercept=prof[i], linetype="dashed")

    return p


def expected_profit_plot(df, rvar, lev, pred, cost=1, margin=2, scale=1, contact=True, prn=True):
    """
    Plot an expected profit curve

    Parameters
    ----------
    df : Polars dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    contact : bool
        Plot a vertical line that shows the optimal contact level. Requires
        that `pred` is a series of probabilities. Values equal to 1 (100% contact)
        will not be plotted
    prn : bool
        Print the maximum expect profit value (default is True, requires contact=True)

    Returns
    -------
    plotnine ggplot object
        Plot of expected profits per quantile

    Examples
    --------
    expected_profit_plot(df, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    expected_profit_plot(df, "buyer", "yes", ["pred_a", "pred_b"], cost=0.5, margin=6)
    dct = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    expected_profit_plot(dct, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    """
    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)
    has_groups = len(pred) > 1 or len(dct.keys()) > 1

    def calc_exp_profit(df_in, pred_col, cost, margin):
        df_in = check_dataframe(df_in)
        prediction = df_in.select(pl.col(pred_col).sort(descending=True))[pred_col]
        profit_vals = prediction.to_numpy() * margin - cost
        n = df_in.height
        return pl.DataFrame(
            {
                "cum_prop": np.arange(1, n + 1) / n,
                "cum_profit": np.cumsum(profit_vals) * scale,
            }
        )

    plot_df = [
        calc_exp_profit(dct[k], p, cost, margin).with_columns(
            pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("predictor")
        )
        for k in dct.keys()
        for p in pred
    ]
    plot_df = pl.concat(plot_df)

    if has_groups:
        p = (
            ggplot(plot_df, aes(x="cum_prop", y="cum_profit", color="predictor"))
            + geom_line()
            + geom_point()
        )
    else:
        p = ggplot(plot_df, aes(x="cum_prop", y="cum_profit")) + geom_line() + geom_point()

    p = (
        p
        + geom_hline(yintercept=0, linetype="dashed")
        + scale_x_continuous(labels=lambda x: [f"{v:.0%}" for v in x])
        + scale_y_continuous(labels=lambda y: [f"{int(v):,}" for v in y])
        + labs(x="Percentage of population targeted", y="Expected Profit", color="")
        + theme_bw()
    )

    if contact:
        cnf = [
            confusion(dct[k], rvar, lev, pr, cost=cost, margin=margin)[-1]
            for k in dct.keys()
            for pr in pred
        ]
        eprof = (
            plot_df.group_by("predictor", maintain_order=True)
            .agg(pl.col("cum_profit").max())["cum_profit"]
            .to_list()
        )
        if prn:
            print(eprof)
        # Add vertical and horizontal lines for contact levels < 1
        for i, cnf_val in enumerate(filter(lambda x: x < 1, cnf)):
            p = p + geom_vline(xintercept=cnf_val, linetype="dashed")
            p = p + geom_hline(yintercept=eprof[i], linetype="dashed")

    return p


def ROME_plot(df, rvar, lev, pred, qnt=10, cost=1, margin=2):
    """
    Plot a ROME curve

    Parameters
    ----------
    df : Polars dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action

    Returns
    -------
    plotnine ggplot object
        Plot of ROME per quantile

    Examples
    --------
    ROME_plot(df, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    ROME_plot(df, "buyer", "yes", ["pred_a", "pred_b"], cost=0.5, margin=6)
    dct = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    ROME_plot(dct, "buyer", "yes", "pred_a", cost=0.5, margin=6)
    """
    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)
    has_groups = len(pred) > 1 or len(dct.keys()) > 1

    rd = [
        ROME_tab(dct[k], rvar, lev, p, qnt=qnt, cost=cost, margin=margin).with_columns(
            pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("predictor")
        )
        for k in dct.keys()
        for p in pred
    ]
    plot_df = pl.concat(rd)

    if has_groups:
        p = (
            ggplot(plot_df, aes(x="cum_prop", y="ROME", color="predictor"))
            + geom_line()
            + geom_point()
        )
    else:
        p = ggplot(plot_df, aes(x="cum_prop", y="ROME")) + geom_line() + geom_point()

    p = (
        p
        + geom_hline(yintercept=0, linetype="dashed")
        + scale_x_continuous(labels=lambda x: [f"{v:.0%}" for v in x])
        + labs(
            x="Percentage of population targeted",
            y="Return on Marketing Expenditures (ROME)",
            color="",
        )
        + theme_bw()
    )

    return p


def gains_plot(df, rvar, lev, pred, qnt=10):
    """
    Plot a cumulative gains curve

    Parameters
    ----------
    df : Polars dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column in df with model predictions
    qnt : int
        Number of quantiles to create

    Returns
    -------
    plotnine ggplot object
        Plot of gains per quantile

    Examples
    --------
    gains_plot(df, "buyer", "yes", "pred_a")
    gains_plot(df, "buyer", "yes", ["pred_a", "pred_b"], qnt=20)
    dct = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    gains_plot(dct, "buyer", "yes", "pred_a")
    """
    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)
    has_groups = len(pred) > 1 or len(dct.keys()) > 1

    rd = [
        gains_tab(dct[k], rvar, lev, p, qnt=qnt).with_columns(
            pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("predictor")
        )
        for k in dct.keys()
        for p in pred
    ]
    plot_df = pl.concat(rd)

    # Baseline diagonal line
    baseline = pl.DataFrame({"cum_prop": [0.0, 1.0], "cum_gains": [0.0, 1.0]})

    if has_groups:
        p = (
            ggplot(plot_df, aes(x="cum_prop", y="cum_gains", color="predictor"))
            + geom_line()
            + geom_point()
        )
    else:
        p = ggplot(plot_df, aes(x="cum_prop", y="cum_gains")) + geom_line() + geom_point()

    p = (
        p
        + geom_line(
            data=baseline,
            mapping=aes(x="cum_prop", y="cum_gains"),
            linetype="dashed",
            color="steelblue",
            inherit_aes=False,
        )
        + scale_x_continuous(labels=lambda x: [f"{v:.0%}" for v in x])
        + scale_y_continuous(labels=lambda y: [f"{v:.0%}" for v in y])
        + labs(x="Percentage of population targeted", y="Percentage Buyers", color="")
        + theme_bw()
    )

    return p


def lift_plot(df, rvar, lev, pred, qnt=10):
    """
    Plot a cumulative lift chart

    Parameters
    ----------
    df : Polars dataframe or a dictionary of dataframes with keys to show multiple curves for different models or data samples
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name, or list, of the column(s) in df with model predictions
    qnt : int
        Number of quantiles to create

    Returns
    -------
    plotnine ggplot object
        Plot of lift per quantile

    Examples
    --------
    lift_plot(df, "buyer", "yes", "pred_a")
    lift_plot(df, "buyer", "yes", ["pred_a", "pred_b"], qnt=20)
    lift = {"Training": df.query("training == 1"), "Test": df.query("training == 0")}
    lift_plot(dct, "buyer", "yes", "pred_a")
    """
    dct = ifelse(isinstance(df, dict), df, {"": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)
    has_groups = len(pred) > 1 or len(dct.keys()) > 1

    rd = [
        lift_tab(dct[k], rvar, lev, p, qnt=qnt).with_columns(
            pl.lit(p + ifelse(k == "", k, f" ({k})")).alias("predictor")
        )
        for k in dct.keys()
        for p in pred
    ]
    plot_df = pl.concat(rd)

    if has_groups:
        p = (
            ggplot(plot_df, aes(x="cum_prop", y="cum_lift", color="predictor"))
            + geom_line()
            + geom_point()
        )
    else:
        p = ggplot(plot_df, aes(x="cum_prop", y="cum_lift")) + geom_line() + geom_point()

    p = (
        p
        + geom_hline(yintercept=1, linetype="dashed")
        + scale_x_continuous(labels=lambda x: [f"{v:.0%}" for v in x])
        + labs(x="Percentage of population targeted", y="Cumulative lift", color="")
        + theme_bw()
    )

    return p


def evalbin(df, rvar, lev, pred, cost=1, margin=2, scale=1, dec=3):
    """
    Evaluate binary classification models. Calculates TP, FP, TN, FN, contact, total,
    TPR, TNR, precision, Fscore, accuracy, profit, ROME, AUC, kappa, and profit index

    Parameters
    ----------
    df : Pandas or Polars dataframe or a dictionary of dataframes with keys to show
        results for multiple model predictions and datasets (training and test)
    rvar : str
        Name of the response variable column in df
    lev : str
        Name of the 'success' level in rvar
    pred : str
        Name of the column, of list of column names, in df with model predictions
    cost : int
        Cost of an action
    margin : int
        Benefit of an action if a successful outcome results from the action
    dec : int
        Number of decimal places to use in rounding

    Examples
    --------
    """
    dct = ifelse(isinstance(df, dict), df, {"All": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)

    def calculate_metrics(key, dfm, pm):
        dfm = check_dataframe(dfm)
        TP, FP, TN, FN, contact = confusion(dfm, rvar, lev, pm, cost, margin)
        total = TN + FN + FP + TP
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        precision = TP / (TP + FP)
        profit = margin * TP - cost * (TP + FP)

        rvar_arr = dfm[rvar].to_numpy()
        pm_arr = dfm[pm].to_numpy()
        fpr, tpr, thresholds = metrics.roc_curve(rvar_arr, pm_arr, pos_label=lev)
        break_even = cost / margin
        gtbe = pm_arr > break_even
        pos = rvar_arr == lev
        if (cost * (TP + FP)) > 0:
            ROME = profit / (cost * (TP + FP))
        else:
            ROME = np.nan

        return pl.DataFrame(
            {
                "Type": [key],
                "predictor": [pm],
                "TP": [TP],
                "FP": [FP],
                "TN": [TN],
                "FN": [FN],
                "total": [total],
                "TPR": [TPR],
                "TNR": [TNR],
                "precision": [precision],
                "Fscore": [2 * (precision * TPR) / (precision + TPR)],
                "accuracy": [(TP + TN) / total],
                "kappa": [metrics.cohen_kappa_score(pos, gtbe)],
                "profit": [profit * scale],
                "index": [0.0],
                "ROME": [ROME],
                "contact": [contact],
                "AUC": [metrics.auc(fpr, tpr)],
            }
        )

    results = []
    for key, val in dct.items():
        for p in pred:
            results.append(calculate_metrics(key, val, p))

    result = pl.concat(results)
    # Calculate index as profit / max profit per Type group
    result = result.with_columns(
        (pl.col("profit") / pl.col("profit").max().over("Type")).alias("index")
    )
    # Round numeric columns and convert to pandas for backward compatibility
    numeric_cols = [c for c in result.columns if result[c].dtype in [pl.Float64, pl.Int64]]
    result = result.with_columns([pl.col(c).round(dec) for c in numeric_cols])
    return result


def auc(rvar, pred, lev=1, weights=None):
    """
    Calculate area under the RO curve (AUC)

    Calculation adapted from https://stackoverflow.com/a/50202118/1974918

    Parameters
    ----------
    rvar : Polars series, Pandas series, or numpy vector
        Vector with the response variable
    pred : Polars series, Pandas series, or numpy vector
        Vector with model predictions
    lev : str
        Name of the 'success' level in rvar

    Returns
    -------
    float :
        AUC metric

    Examples
    --------
    auc(dvd.buy, np.random.uniform(size=20000), "yes")
    auc(dvd.buy, rsm.ifelse(dvd.buy == "yes", 1, 0), "yes")
    """
    # Convert inputs to polars Series
    if not isinstance(rvar, pl.Series):
        rvar = pl.Series("rvar", rvar)
    if not isinstance(pred, pl.Series):
        pred = pl.Series("pred", pred)

    # Create boolean mask for success level
    rvar_bool = (rvar == lev) if lev is not None else rvar.cast(pl.Boolean)

    if weights is None:
        df = pl.DataFrame({"pred": pred, "rvar": rvar_bool})
        df = df.with_columns(pl.col("pred").rank(method="average").alias("rank"))
        rd = df.filter(~pl.col("rvar"))["rank"].sum()
        n1 = (~rvar_bool).sum()
        n2 = rvar_bool.sum()
    else:
        if not isinstance(weights, pl.Series):
            weights = pl.Series("weights", weights)
        df = table2data(
            pl.DataFrame({"pred": pred, "rvar": rvar_bool, "weights": weights}), "weights"
        )
        df = df.with_columns(pl.col("pred").rank(method="average").alias("rank"))
        rd = df.filter(~pl.col("rvar"))["rank"].sum()
        # For weighted case, sum weights by rvar
        weight_df = pl.DataFrame({"rvar": rvar_bool, "weights": weights})
        n1 = weight_df.filter(~pl.col("rvar"))["weights"].sum()
        n2 = weight_df.filter(pl.col("rvar"))["weights"].sum()

    U = rd - n1 * (n1 + 1) / 2
    wt = U / n1 / n2
    return wt if wt >= 0.5 else 1 - wt
