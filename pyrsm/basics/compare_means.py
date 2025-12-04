from typing import Literal

import numpy as np
import polars as pl
from plotnine import (
    aes,
    geom_boxplot,
    geom_col,
    geom_density,
    geom_jitter,
    geom_segment,
    ggplot,
    labs,
    theme_bw,
)
from scipy import stats
from statsmodels.stats import multitest

import pyrsm.basics.plotting_utils as pu
import pyrsm.basics.utils as bu
import pyrsm.basics.display_utils as du
import pyrsm.radiant.utils as ru
from pyrsm.model.model import sig_stars
from pyrsm.utils import check_dataframe, ifelse


class compare_means:
    """
    Compare group means for polars inputs with paired/independent samples,
    adjustment options, and quick plotting.

    Quick start
    -----------
    >>> import pyrsm as rsm
    >>> salary, salary_description = rsm.load_data(pkg="basics", name="salary")
    >>> cm = rsm.basics.compare_means({"salary": salary}, var1="rank", var2="salary")
    >>> cm.summary()
    >>> cm.plot()  # returns (Figure, Axes)

    >>> salary, _ = rsm.load_data(pkg="basics", name="salary")
    >>> rsm.basics.compare_means(salary, var1="sex", var2="salary", test_type="wilcox")

    See the worked notebook (UI + code) in examples/basics/basics-compare-means.ipynb.

    Parameters
    ----------
    data : pl.DataFrame | dict[str, pl.DataFrame]
        Input data as a DataFrame or a dictionary where the first key becomes the
        name stored on the result.
    var1 : str
        Grouping variable. If numeric and var2 is a list, the data are melted to
        long form so each numeric column becomes a group level.
    var2 : str | list[str] | tuple[str, ...]
        Numeric variable(s) to compare across groups. When var1 is categorical, one
        numeric column is expected; when var1 is numeric, each entry in the list
        becomes a group to compare.
    comb : list[str]
        Optional custom comparisons (e.g., ["a:b", "a:c"]). Defaults to all pairwise
        combinations of var1 levels.
    alt_hyp : Literal["two-sided", "greater", "less"]
        Alternative hypothesis used for test statistic and confidence labels.
    conf : float
        Confidence level for intervals.
    sample_type : Literal["independent", "paired"]
        Indicates paired vs independent samples.
    adjust : Literal[None, "bonferroni"]
        Adjustment for multiple testing using statsmodels.multitest when set.
    test_type : Literal["t-test", "wilcox"]
        Statistical test to run (Welch t-test by default or Wilcoxon paths).

    Attributes
    ----------
    data : pl.DataFrame
        Polars version of the provided data.
    var1 : str
        Grouping variable used after any melt operation.
    var2 : str | list[str]
        Numeric variable(s) used after any melt operation.
    comb : list[str]
        Level combinations evaluated.
    descriptive_stats : pl.DataFrame
        Per-group summary with mean, n, n_missing, sd, se, and margin of error.
    comp_stats : pl.DataFrame
        Pairwise comparisons with diffs, p-values (optionally adjusted), t values,
        degrees of freedom, confidence interval bounds, and significance stars.
    levels : list[str]
        Category levels for var1 after casting to categorical.
    name : str
        Dataset label, taken from the dictionary key when provided.

    Notes
    -----
    The `summary` method prints a compact or expanded view (`extra=True`). The
    `plot` method returns a plotnine ggplot object.
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        var1: str,
        var2: str,
        comb: list[str] = [],
        alt_hyp: Literal["two-sided", "greater", "less"] = "two-sided",
        conf: float = 0.95,
        sample_type: Literal["independent", "paired"] = "independent",
        adjust: Literal[None, "bonferroni"] = None,
        test_type: Literal["t-test", "wilcox"] = "t-test",
    ):
        """
        Constructs all the necessary attributes for the compare_means object.

        Parameters
        ----------
        data : pl.DataFrame | dict[str, pl.DataFrame]
            The input data for the hypothesis test as a Polars DataFrame. If a dictionary is provided, the key should be the name of the dataframe.
        var1 : str
            The first variable/column name to include in the test. This variable can be numeric or categorical. If it is categorical, the hypothesis test will be performed for each level of the variable.
        var2 : str
            The second variable/column name or names to include in the test. These variables must be numeric. If multiple variables are provided, the hypothesis test will be performed for each combination of variables. If var1 is categorical, only one variable can be provided.
        comb : str or list of strings
            Combinations of levels of var1 (e.g., "a:b" or ["a:b", "a:d"]) or combinations of var1 with variables in the var2 list (e.g., "x1:x2" or ["x1:x2", "x1:x4"]). If an empty list is provided (default) all combinations will be evaluated.
        alt_hyp : str, optional
            The alternative hypothesis ('two-sided', 'greater', 'less') (default is 'two-sided').
        conf : float, optional
            The confidence level for the test (default is 0.95).
        sample_type : str
            The type of samples ('independent' or 'paired') (default is 'independent').An example of paired samples is when the same subjects are measured at two different time points.
        adjust : str, optional
            Adjustment for multiple testing (None or 'bonferroni' or other options provided but statsmodels.stats.multitest.multipletests).
        test_type : str
            The type of test ('t-test' or 'wilcox') (default is 't-test'). The key difference between the two tests is that the t-test assumes that the data is normally distributed, while the Wilcoxon test does not.
        """
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)
        self.var1 = var1
        self.var2 = var2

        var1_series = self.data.get_column(self.var1)
        if var1_series.dtype.is_numeric():
            var2_list = ifelse(
                isinstance(var2, str),
                [var2],
                ifelse(isinstance(var2, tuple), list(var2), list(var2)),
            )
            for v in var2_list:
                if not self.data.get_column(v).dtype.is_numeric():
                    raise Exception(f"Variable {v} is not numeric.")

            if (
                len(var2_list) > 1
                or len([v for v in comb if self.var1 in v]) > 0
                or self.data.get_column(self.var1).n_unique() > 10
            ):
                cols = [self.var1] + var2_list
                self.data = self.data.select(cols).unpivot(
                    index=[],
                    on=cols,
                    variable_name="variable",
                    value_name="value",
                )
                self.var1 = "variable"
                self.var2 = "value"
        else:
            var2_list = ifelse(isinstance(var2, str), [var2], list(var2))

        self.data = self.data.with_columns(pl.col(self.var1).cast(pl.Categorical))
        self.levels = (
            self.data.select(pl.col(self.var1).unique(maintain_order=True)).to_series().to_list()
        )

        if len(comb) == 0:
            self.comb = ru.iterms(self.levels)
        else:
            self.comb = ifelse(isinstance(comb, str), [comb], comb)

        self.alt_hyp = alt_hyp
        self.conf = conf
        self.alpha = 1 - self.conf
        self.sample_type = sample_type
        self.adjust = adjust
        self.test_type = test_type

        def welch_dof(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
            # stats.ttest_ind uses Welch's t-test when equal_var=False
            # but does not return the degrees of freedom
            x = x_vals
            y = y_vals
            if x.size == 0 or y.size == 0:  # address division by zero
                return np.nan
            x_var = np.nanvar(x, ddof=1)
            y_var = np.nanvar(y, ddof=1)
            dof = (x_var / x.size + y_var / y.size) ** 2 / (
                (x_var / x.size) ** 2 / (x.size - 1) + (y_var / y.size) ** 2 / (y.size - 1)
            )

            return dof

        def margin_of_error(n: float, se: float) -> float:
            if n is None or se is None or np.isnan(n) or np.isnan(se) or n <= 1:
                return np.nan
            tscore = stats.t.ppf((1 + self.conf) / 2, n - 1)
            return (tscore * se).real

        descriptive_stats = (
            self.data.group_by(self.var1, maintain_order=True)
            .agg(
                pl.col(self.var2).mean().alias("mean"),
                pl.len().alias("n_total"),
                pl.col(self.var2).null_count().alias("n_missing"),
                pl.col(self.var2).std(ddof=1).alias("sd"),
            )
            .with_columns((pl.col("n_total") - pl.col("n_missing")).alias("n"))
            .with_columns(
                pl.when(pl.col("n") > 0)
                .then(pl.col("sd") / pl.col("n").sqrt())
                .otherwise(pl.lit(np.nan))
                .alias("se")
            )
            .with_columns(
                pl.struct(["n", "se"])
                .map_elements(lambda s: margin_of_error(s["n"], s["se"]), return_dtype=pl.Float64)
                .alias("me")
            )
        )

        self.descriptive_stats = descriptive_stats.select(
            [self.var1, "mean", "n", "n_missing", "sd", "se", "me"]
        )

        if self.alt_hyp == "less":
            alt_hyp_sign = "less than"
        elif self.alt_hyp == "two-sided":
            alt_hyp_sign = "not equal to"
        else:
            alt_hyp_sign = "greater than"

        comp_stats = []
        for c in self.comb:
            v1, v2 = c.split(":")
            null_hyp = f"{v1} = {v2}"
            alt_hyp = f"{v1} {alt_hyp_sign} {v2}"

            # Replace NaN with null, then compute the mean (ignoring nulls)
            x_series = self.data.filter(pl.col(self.var1) == v1).get_column(self.var2).drop_nulls()
            y_series = self.data.filter(pl.col(self.var1) == v2).get_column(self.var2).drop_nulls()
            diff = x_series.mean() - y_series.mean()
            x = x_series.to_numpy()
            y = y_series.to_numpy()

            if x.size != y.size and self.sample_type == "paired":
                raise ValueError(
                    """The two samples must have the same size for a paired
                    sample test. Choose independent samples instead."""
                )

            if self.test_type == "t-test":
                if self.sample_type == "independent":
                    result = stats.ttest_ind(
                        x,
                        y,
                        equal_var=False,
                        nan_policy="omit",
                        alternative=self.alt_hyp,
                    )
                else:
                    result = stats.ttest_rel(x, y, nan_policy="omit", alternative=self.alt_hyp)
            elif self.test_type == "wilcox":
                if self.sample_type == "independent":
                    result = stats.ranksums(x, y, alternative=self.alt_hyp)
                else:
                    result = stats.wilcoxon(x, y, correction=True, alternative=self.alt_hyp)

            t_val, p_val = result.statistic, result.pvalue
            se = diff / t_val if t_val != 0 else np.inf
            df = welch_dof(x, y)

            if self.alt_hyp == "two-sided":
                tscore = stats.t.ppf((1 + self.conf) / 2, df)
            else:
                tscore = stats.t.ppf(self.conf, df)
            me = (tscore * se).real

            if self.alt_hyp == "less":
                ci = [-np.inf, diff + me]
            elif self.alt_hyp == "two-sided":
                ci = [diff - me, diff + me]
            else:
                ci = [diff - me, np.inf]

            comp_stats.append(
                [
                    null_hyp,
                    alt_hyp,
                    diff,
                    p_val,
                    se,
                    t_val,
                    df,
                    ci[0],
                    ci[1],
                    sig_stars([p_val])[0],
                ]
            )

        cl = bu.ci_label(self.alt_hyp, self.conf)
        self.comp_stats = pl.DataFrame(
            comp_stats,
            schema=[
                ("Null hyp.", pl.Utf8),
                ("Alt. hyp.", pl.Utf8),
                ("diff", pl.Float64),
                ("p.value", pl.Float64),
                ("se", pl.Float64),
                ("t.value", pl.Float64),
                ("df", pl.Float64),
                (cl[0], pl.Float64),
                (cl[1], pl.Float64),
                ("", pl.Utf8),
            ],
            orient="row",
        )

        if self.adjust is not None:
            alpha = self.alpha if self.alt_hyp == "two-sided" else self.alpha * 2
            adjusted = multitest.multipletests(
                self.comp_stats["p.value"].to_numpy(), method=self.adjust, alpha=alpha
            )[1]
            self.comp_stats = self.comp_stats.with_columns(
                pl.Series("p.value", adjusted),
                pl.Series("", sig_stars(list(adjusted))),
            )

    def summary(self, extra=False, dec=3) -> None:
        """
        Prints a summary of the hypothesis test.

        Parameters
        ----------
        extra : bool
            Whether to include additional columns in the output (default is False).
        dec : int, optional
            The number of decimal places to display (default is 3).
        """
        display = du.SummaryDisplay(
            header_func=self._summary_header,
            plain_func=self._summary_plain,
            styled_func=self._style_tables,
        )
        display.display(extra=extra, dec=dec)

    def _summary_header(self):
        """Print the summary header."""
        print(f"Pairwise mean comparisons ({self.test_type})")
        print(f"Data      : {self.name}")
        print(f"Variables : {self.var1}, {self.var2}")
        print(f"Samples   : {self.sample_type}")
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.adjust}\n")

    def _summary_plain(self, extra=False, dec=3):
        """Print plain text tables using polars Config."""
        desc = self.descriptive_stats.with_columns(pl.col(["mean", "sd", "se", "me"]).round(dec))

        comp_stats = self.comp_stats
        if not extra:
            comp_stats = comp_stats.select(["Null hyp.", "Alt. hyp.", "diff", "p.value", ""])

        pvals = comp_stats["p.value"].to_list()
        formatted = [ifelse(p < 0.001, "< .001", f"{round(p, dec)}") for p in pvals]
        comp_stats = comp_stats.with_columns(pl.Series("p.value", formatted, dtype=pl.Utf8))

        # Round all numeric columns
        numeric_cols = [c for c in comp_stats.columns if comp_stats[c].dtype.is_float()]
        if numeric_cols:
            comp_stats = comp_stats.with_columns(pl.col(numeric_cols).round(dec))

        with pl.Config(
            tbl_rows=-1,
            tbl_cols=-1,
            fmt_str_lengths=100,
            tbl_width_chars=200,
            tbl_hide_dataframe_shape=True,
            tbl_hide_column_data_types=True,
            tbl_hide_dtype_separator=True,
        ):
            print(desc)
            print(comp_stats)

    def _style_tables(self, extra=False, dec=3):
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display, HTML

        desc = self.descriptive_stats

        comp_stats = self.comp_stats
        if not extra:
            comp_stats = comp_stats.select(["Null hyp.", "Alt. hyp.", "diff", "p.value", ""])

        pvals = comp_stats["p.value"].to_list()
        formatted = [ifelse(p < 0.001, "< .001", f"{round(p, dec)}") for p in pvals]
        comp_stats = comp_stats.with_columns(pl.Series("p.value", formatted, dtype=pl.Utf8))

        desc_gt = (
            desc.style.tab_header(
                title="Descriptive Statistics",
                subtitle=f"Variable: {self.var2} by {self.var1}",
            )
            .fmt_number(columns=["mean", "sd", "se", "me"], decimals=dec, use_seps=False)
            .fmt_integer(columns=["n", "n_missing"], use_seps=False)
            .tab_options(table_margin_left="0px")
        )

        if extra:
            ci_cols = [c for c in comp_stats.columns if "%" in c]
            comp_gt = (
                comp_stats.style.tab_header(
                    title=f"Pairwise Comparisons ({self.test_type})",
                    subtitle=f"Samples: {self.sample_type}, Confidence: {self.conf}",
                )
                .fmt_number(
                    columns=["diff", "se", "t.value", "df"] + ci_cols, decimals=dec, use_seps=False
                )
                .tab_options(table_margin_left="0px")
            )
        else:
            comp_gt = (
                comp_stats.style.tab_header(
                    title=f"Pairwise Comparisons ({self.test_type})",
                    subtitle=f"Samples: {self.sample_type}, Confidence: {self.conf}",
                )
                .fmt_number(columns=["diff"], decimals=dec, use_seps=False)
                .tab_options(table_margin_left="0px")
            )

        display(desc_gt)
        display(comp_gt)

    def style(self, extra=False, dec=3):
        """
        Display styled DataFrames in notebook using great_tables.

        Parameters
        ----------
        extra : bool
            Whether to include additional columns in the output (default is False).
        dec : int, optional
            The number of decimal places to display (default is 3).
        """
        self._summary_header()
        self._style_tables(extra=extra, dec=dec)
        du.print_sig_codes()

    def plot(
        self, plots: Literal["scatter", "box", "density", "bar"] = "scatter", nobs: int = None
    ):
        """
        Plots the results of the hypothesis test.

        Parameters
        ----------
        plots : str
            The type of plot to create ('scatter', 'box', 'density', 'bar').
        nobs : int, optional
            The number of observations to plot (default is None in which case all available data points will be used).

        Returns
        -------
        plotnine.ggplot
            A plotnine ggplot object that can be displayed, saved, or composed.
        """
        data = self.data.drop_nulls(subset=[self.var2])
        if nobs is not None and nobs != np.inf and nobs != -1 and nobs < data.height:
            data = data.sample(nobs)

        if plots == "scatter":
            from plotnine import stat_summary
            p = (
                ggplot(data, aes(x=self.var1, y=self.var2))
                + geom_jitter(width=0.2, alpha=0.5, color=pu.PlotConfig.FILL)
                + stat_summary(
                    fun_y=np.mean,
                    fun_ymin=np.mean,
                    fun_ymax=np.mean,
                    geom="crossbar",
                    color="blue",
                    linetype="dashed",
                    width=0.5,
                    size=0.8,
                    fatten=0,
                )
                + labs(x=self.var1, y=self.var2)
                + theme_bw()
            )

        elif plots == "box":
            p = (
                ggplot(data, aes(x=self.var1, y=self.var2))
                + geom_boxplot(fill=pu.PlotConfig.FILL, alpha=0.7)
                + labs(x=self.var1, y=self.var2)
                + theme_bw()
            )

        elif plots == "density":
            p = (
                ggplot(data, aes(x=self.var2, color=self.var1, fill=self.var1))
                + geom_density(alpha=0.3)
                + labs(x=self.var2, y="Density", color=self.var1, fill=self.var1)
                + theme_bw()
            )

        elif plots == "bar":
            # Calculate means per group
            means = data.group_by(self.var1).agg(
                pl.col(self.var2).mean().alias("mean")
            )
            p = (
                ggplot(means, aes(x=self.var1, y="mean"))
                + geom_col(fill=pu.PlotConfig.FILL, alpha=0.8)
                + labs(x=self.var1, y=f"Mean {self.var2}")
                + theme_bw()
            )

        else:
            raise ValueError(f"Invalid plot type: {plots}. Choose from 'scatter', 'box', 'density', 'bar'.")

        return p
