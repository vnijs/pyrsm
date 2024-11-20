import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from pyrsm.model import sig_stars
from pyrsm.utils import ifelse, check_dataframe
from typing import Literal
from statsmodels.stats import multitest
import pyrsm.basics.utils as bu
import pyrsm.radiant.utils as ru


class compare_means:
    """
    A class to perform comparison of means hypothesis testing. See the notebook 
    linked below for a worked example, including the web UI:

    https://github.com/vnijs/pyrsm/blob/main/examples/basics-compare-means.ipynb

    Attributes
    ----------
    data : pd.DataFrame | pl.DataFrame
        The input data for the hypothesis test as a Pandas or Polars DataFrame.
    var1 : str
        The first variable/column name to test.
    var2 : str
        The second variable/column name to test.
    alt_hyp : str
        The alternative hypothesis ('two-sided', 'greater', 'less').
    conf : float
        The confidence level for the test.
    comp_value : float
        The comparison value for the test.
    t_val : float
        The t-statistic value.
    p_val : float
        The p-value of the test.
    ci : tuple
        The confidence interval of the test.
    mean1 : float
        The mean of the first variable.
    mean2 : float
        The mean of the second variable.
    n1 : int
        The number of observations for the first variable.
    n2 : int
        The number of observations for the second variable.
    sd1 : float
        The standard deviation of the first variable.
    sd2 : float
        The standard deviation of the second variable.
    se : float
        The standard error of the difference between the means.
    me : float
        The margin of error.
    diff : float
        The difference between the means of the two variables.
    df : int
        The degrees of freedom.

    Methods
    -------
    __init__(data, var1, var2, alt_hyp='two-sided', conf=0.95, comp_value=0)
        Initializes the compare_means class with the provided data and parameters.
    summary(dec=3)
        Prints a summary of the hypothesis test.
    plot()
        Plots the results of the hypothesis test.

    Examples
    --------

    import pandas as pd
    import pyrsm as rsm
    salary, salary_description = rsm.load_data(pkg="basics", name="salary")
    cm = rsm.basics.compare_means({"salary": salary}, var1="rank", var2="salary", alt_hyp="less")
    cm.summary()
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        var1: str,
        var2: str,
        comb: list[tuple[str, str]] = [],
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
        data : pd.DataFrame | pl.DataFrame
            The input data for the hypothesis test as a Pandas or Polars DataFrame. If a dictionary is provided, the key should be the name of the dataframe.
        var1 : str
            The first variable/column name to include in the test. This variable can be numeric or categorical. If it is categorical, the hypothesis test will be performed for each level of the variable.
        var2 : str
            The second variable/column name or names to include in the test. These variables must be numeric. If multiple variables are provided, the hypothesis test will be performed for each combination of variables. If var1 is categorical, only one variable can be provided.
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
        self.comb = comb

        var1_series = self.data[self.var1]
        if pd.api.types.is_numeric_dtype(var1_series):
            var2 = ifelse(
                isinstance(var2, str),
                [var2],
                ifelse(isinstance(var2, tuple), list(var2), var2),
            )
            if (
                len(self.var2) > 1
                or len([v for v in self.comb if self.var1 in v]) > 0
                or var1_series.nunique() > 10
            ):
                self.data = self.data.loc[:, [self.var1] + var2].melt()
                self.var1 = "variable"
                self.var2 = "value"

        self.data[self.var1] = self.data[self.var1].astype("category")
        self.levels = list(self.data[self.var1].cat.categories)

        if len(self.comb) == 0:
            self.comb = ru.iterms(self.levels)

        self.alt_hyp = alt_hyp
        self.conf = conf
        self.alpha = 1 - self.conf
        self.sample_type = sample_type
        self.adjust = adjust
        self.test_type = test_type

        def welch_dof(v1: str, v2: str) -> float:
            # stats.ttest_ind uses Welch's t-test when equal_var=False
            # but does not return the degrees of freedom
            x = self.data.loc[self.data[self.var1] == v1, self.var2]
            y = self.data.loc[self.data[self.var1] == v2, self.var2]
            if x.size == 0 or y.size == 0:  # address division by zero
                return np.nan
            dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
                (x.var() / x.size) ** 2 / (x.size - 1)
                + (y.var() / y.size) ** 2 / (y.size - 1)
            )

            return dof

        descriptive_stats = []
        for lev in self.levels:
            subset = self.data.loc[self.data[self.var1] == lev, self.var2]
            mean = np.nanmean(subset)
            n_missing = subset.isna().sum()
            n = len(subset) - n_missing
            sd = subset.std()
            se = subset.sem()
            tscore = stats.t.ppf((1 + self.conf) / 2, n - 1)
            me = (tscore * se).real
            row = [lev, mean, n, n_missing, sd, se, me]
            descriptive_stats.append(row)

        self.descriptive_stats = pd.DataFrame(
            descriptive_stats,
            columns=[self.var1, "mean", "n", "n_missing", "sd", "se", "me"],
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
            diff = np.nanmean(
                self.data[self.data[self.var1] == v1][self.var2]
            ) - np.nanmean(self.data[self.data[self.var1] == v2][self.var2])

            x = self.data.loc[self.data[self.var1] == v1, self.var2]
            y = self.data.loc[self.data[self.var1] == v2, self.var2]
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
                    result = stats.ttest_rel(
                        x, y, nan_policy="omit", alternative=self.alt_hyp
                    )
            elif self.test_type == "wilcox":
                if self.sample_type == "independent":
                    result = stats.ranksums(x, y, alternative=self.alt_hyp)
                else:
                    result = stats.wilcoxon(
                        x, y, correction=True, alternative=self.alt_hyp
                    )

            t_val, p_val = result.statistic, result.pvalue
            se = diff / t_val
            df = welch_dof(v1, v2)

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
        self.comp_stats = pd.DataFrame(
            comp_stats,
            columns=[
                "Null hyp.",
                "Alt. hyp.",
                "diff",
                "p.value",
                "se",
                "t.value",
                "df",
                cl[0],
                cl[1],
                "",
            ],
        )

        if self.adjust is not None:
            if self.alt_hyp == "two-sided":
                alpha = self.alpha
            else:
                alpha = self.alpha * 2
            self.comp_stats["p.value"] = multitest.multipletests(
                self.comp_stats["p.value"], method=self.adjust, alpha=alpha
            )[1]
            self.comp_stats[""] = sig_stars(self.comp_stats["p.value"])

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
        print(f"Pairwise mean comparisons ({self.test_type})")
        print(f"Data      : {self.name}")
        print(f"Variables : {self.var1}, {self.var2}")
        print(f"Samples   : {self.sample_type}")
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.adjust}")

        print(self.descriptive_stats.round(dec).to_string(index=False))

        comp_stats = self.comp_stats.copy()
        if not extra:
            comp_stats = comp_stats.iloc[:, [0, 1, 2, 3, -1]]

        comp_stats["p.value"] = ifelse(
            comp_stats["p.value"] < 0.001, "< .001", round(comp_stats["p.value"], dec)
        )
        print(comp_stats.round(dec).to_string(index=False))
        print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    def plot(self, plots: Literal["scatter", "box", "density", "bar"] = "scatter", nobs: int = None) -> None:
        """
        Plots the results of the hypothesis test.

        Parameters
        ----------
        plots : str
            The type of plot to create ('scatter', 'box', 'density', 'bar').
        nobs : int, optional
            The number of observations to plot (default is None in which case all available data points will be used).
        """
        if plots == "scatter":
            if (
                nobs is not None
                and nobs < self.data.shape[0]
                and nobs != np.inf
                and nobs != -1
            ):
                data = self.data.copy().sample(nobs)
            else:
                data = self.data.copy()

            sns.swarmplot(data=data, x=self.var1, y=self.var2, alpha=0.5)

            # Get the unique categories and their indices
            categories = data[self.var1].cat.categories
            category_indices = {category: i for i, category in enumerate(categories)}

            category_means = data.groupby(self.var1, observed=False)[self.var2].mean()

            # Add a horizontal line for each category at the mean of the value for that category
            for category, mean in category_means.items():
                plt.hlines(
                    y=mean,
                    xmin=category_indices[category] - 0.3,
                    xmax=category_indices[category] + 0.3,
                    color="blue",
                    linestyle="--",
                    linewidth=2,
                    zorder=2,
                )

        elif plots == "box":
            sns.boxplot(data=self.data, x=self.var1, y=self.var2)
        elif plots == "density":
            sns.kdeplot(data=self.data, x=self.var2, hue=self.var1)
        elif plots == "bar":
            sns.barplot(
                data=self.data,
                y=self.var2,
                x=self.var1,
                # yerr=self.descriptive_stats["se"], # shapes don't align for some reason
            )
        else:
            print("Invalid plot type")
