from cmath import sqrt
from typing import Union

import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats import multitest
from statsmodels.stats.proportion import proportions_ztest
from plotnine import (
    aes,
    geom_col,
    ggplot,
    ggtitle,
    labs,
    position_dodge,
    theme_bw,
)

import pyrsm.basics.utils as bu
import pyrsm.basics.display_utils as du
import pyrsm.basics.plotting_utils as pu
import pyrsm.radiant.utils as ru
from pyrsm.model.model import sig_stars
from pyrsm.utils import check_dataframe, ifelse


class compare_props:
    """
    Compare proportions across levels of a categorical variable in a Polars
    dataframe. See the notebook linked below for a worked example,
    including the web UI:

    https://github.com/vnijs/pyrsm/blob/main/examples/basics-compare-means.ipynb
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        var1: str,
        var2: str,
        lev: str,
        comb: list[str] = [],
        alt_hyp: str = "two-sided",
        conf: float = 0.95,
        adjust: str = None,
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)
        self.var1 = var1
        self.var2 = var2
        self.lev = lev
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.alpha = 1 - self.conf
        self.adjust = adjust

        # Get unique levels from var1 (cast to string if needed for categorical)
        var1_col = self.data[self.var1]
        if var1_col.dtype == pl.Categorical:
            self.levels = var1_col.cat.get_categories().to_list()
        else:
            self.levels = sorted(self.data[self.var1].unique().to_list())

        if len(comb) == 0:
            self.comb = ru.iterms(self.levels)
        else:
            self.comb = ifelse(isinstance(comb, str), [comb], comb)

        descriptive_stats = []
        for level in self.levels:
            # Filter for rows where var1 == level and var2 == self.lev
            subset = self.data.filter(
                (pl.col(self.var1) == level) & (pl.col(self.var2) == self.lev)
            )
            ns = subset.height

            # Count missing in var2 for this level of var1
            level_data = self.data.filter(pl.col(self.var1) == level)
            n_missing = level_data[self.var2].null_count()
            n = level_data.height - n_missing

            p = ns / n
            sd = sqrt(p * (1 - p)).real
            se = (sd / sqrt(n)).real
            z_score = stats.norm.ppf(1 - self.alpha / 2)
            me = (z_score * sd / sqrt(n)).real
            descriptive_stats.append([level, ns, p, n, n_missing, sd, se, me])

        self.descriptive_stats = pl.DataFrame(
            descriptive_stats,
            schema=[
                self.var1,
                self.lev,
                "p",
                "n",
                "n_missing",
                "sd",
                "se",
                "me",
            ],
            orient="row",
        )

        if self.alt_hyp == "less":
            alt_hyp_sign = "less than"
        elif self.alt_hyp == "two-sided":
            alt_hyp_sign = "not equal to"
        else:
            alt_hyp_sign = "greater than"

        def wald_ci(n1, p1, n2, p2, z):
            # what R uses for a comparison of proportions
            se = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
            diff = p1 - p2
            return [diff - z * se, diff + z * se]

        comp_stats = []
        for c in self.comb:
            v1, v2 = c.split(":")
            null_hyp = f"{v1} = {v2}"
            alt_hyp = f"{v1} {alt_hyp_sign} {v2}"

            # Get data for each level, dropping nulls
            x_data = self.data.filter(pl.col(self.var1) == v1).drop_nulls(self.var2)
            y_data = self.data.filter(pl.col(self.var1) == v2).drop_nulls(self.var2)

            c1 = x_data.filter(pl.col(self.var2) == self.lev).height
            c2 = y_data.filter(pl.col(self.var2) == self.lev).height
            n1 = x_data.height
            n2 = y_data.height
            p1 = c1 / n1
            p2 = c2 / n2
            diff = p1 - p2

            pzt = ifelse(
                self.alt_hyp == "less",
                "smaller",
                ifelse(self.alt_hyp == "greater", "larger", "two-sided"),
            )

            z_val, p_val = proportions_ztest([c1, c2], [n1, n2], alternative=pzt)
            zc = stats.norm.ppf(self.conf)
            if self.alt_hyp == "less":
                ci = [-1, wald_ci(n1, p1, n2, p2, zc)[1]]
            elif self.alt_hyp == "two-sided":
                zc = stats.norm.ppf(1 - (1 - self.conf) / 2)
                ci = wald_ci(n1, p1, n2, p2, zc)
            else:
                ci = [wald_ci(n1, p1, n2, p2, zc)[0], 1]

            comp_stats.append(
                [
                    null_hyp,
                    alt_hyp,
                    diff,
                    p_val,
                    z_val**2,
                    1,
                    ci[0],
                    ci[1],
                    sig_stars([p_val])[0],
                ]
            )

        cl = bu.ci_label(self.alt_hyp, self.conf)
        self.comp_stats = pl.DataFrame(
            comp_stats,
            schema=[
                "Null hyp.",
                "Alt. hyp.",
                "diff",
                "p.value",
                "chisq.value",
                "df",
                cl[0],
                cl[1],
                "",
            ],
            orient="row",
        )

        if self.adjust is not None:
            if self.alt_hyp == "two-sided":
                alpha = self.alpha
            else:
                alpha = self.alpha * 2
            adjusted_pvals = multitest.multipletests(
                self.comp_stats["p.value"].to_list(), method=self.adjust, alpha=alpha
            )[1]
            self.comp_stats = self.comp_stats.with_columns(
                pl.Series("p.value", adjusted_pvals)
            )
            self.comp_stats = self.comp_stats.with_columns(
                pl.Series("", sig_stars(adjusted_pvals))
            )

    def summary(self, extra=False, dec: int = 3) -> None:
        display = du.SummaryDisplay(
            header_func=self._summary_header,
            plain_func=lambda e, d: self._summary_plain(e, d),
            styled_func=lambda e, d: self._style_tables(e, d),
        )
        display.display(extra=extra, dec=dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        print("Pairwise proportion comparisons")
        print(f"Data      : {self.name}")
        print(f"Variables : {self.var1}, {self.var2}")
        print(f'Level     : "{self.lev}" in {self.var2}')
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.adjust}\n")

    def _summary_plain(self, extra: bool = False, dec: int = 3) -> None:
        """Print plain text tables."""
        comp_stats = self.comp_stats.clone()
        if not extra:
            cols = comp_stats.columns
            comp_stats = comp_stats.select([cols[0], cols[1], cols[2], cols[3], cols[-1]])

        # Format p-values
        p_vals = comp_stats["p.value"].to_list()
        p_formatted = [du.format_pval(p, dec) for p in p_vals]
        comp_stats = comp_stats.with_columns(pl.Series("p.value", p_formatted))

        # Round numeric columns for display
        desc_rounded = self.descriptive_stats.select([
            pl.col(self.var1),
            pl.col(self.lev),
            pl.col("p").round(dec),
            pl.col("n"),
            pl.col("n_missing"),
            pl.col("sd").round(dec),
            pl.col("se").round(dec),
            pl.col("me").round(dec),
        ])

        du.print_plain_tables(desc_rounded, comp_stats)

    def _style_tables(self, extra: bool = False, dec: int = 3) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        comp_stats = self.comp_stats.clone()
        if not extra:
            cols = comp_stats.columns
            comp_stats = comp_stats.select([cols[0], cols[1], cols[2], cols[3], cols[-1]])

        # Format p-values
        p_vals = comp_stats["p.value"].to_list()
        p_formatted = [du.format_pval(p, dec) for p in p_vals]
        comp_stats = comp_stats.with_columns(pl.Series("p.value", p_formatted, dtype=pl.Utf8))

        gt1 = du.style_table(
            self.descriptive_stats,
            title="Descriptive Statistics",
            subtitle=f'Level: "{self.lev}" in {self.var2}',
            number_cols=["p", "sd", "se", "me"],
            integer_cols=["n", "n_missing", self.lev],
            dec=dec,
        )

        ci_cols = [c for c in comp_stats.columns if "%" in c]
        number_cols = ["diff"] + (["chisq.value", "se"] if extra else []) + ci_cols

        gt2 = du.style_table(
            comp_stats,
            title="Pairwise Comparisons",
            subtitle=f"Confidence: {self.conf}",
            number_cols=number_cols,
            integer_cols=["df"] if extra else None,
            dec=dec,
        )

        display(gt1)
        display(gt2)

    def plot(self, plots: str = "bar"):
        """
        Plot proportions comparison.

        Parameters
        ----------
        plots : str
            Plot type: "bar" for mean proportions, "dodge" for grouped bar chart

        Returns
        -------
        plotnine.ggplot
            A plotnine ggplot object
        """
        if plots == "bar":
            # Compute mean proportions by var1
            plot_data = self.data.select([self.var1, self.var2]).with_columns(
                pl.when(pl.col(self.var2) == self.lev)
                .then(1.0)
                .otherwise(0.0)
                .alias("proportion")
            )
            means = plot_data.group_by(self.var1).agg(
                pl.col("proportion").mean().alias("mean_prop")
            )
            p = (
                ggplot(means, aes(x=self.var1, y="mean_prop"))
                + geom_col(fill=pu.PlotConfig.FILL, alpha=0.7)
                + labs(x=self.var1, y=f"Proportion ({self.lev})")
                + ggtitle(f'Proportion comparison: "{self.lev}" in {self.var2}')
                + theme_bw()
            )
            return p

        elif plots == "dodge":
            # Compute proportions using polars
            counts = (
                self.data.select([self.var1, self.var2])
                .group_by([self.var1, self.var2])
                .len()
            )
            totals = counts.group_by(self.var1).agg(pl.col("len").sum().alias("total"))
            proportions = counts.join(totals, on=self.var1).with_columns(
                (pl.col("len") / pl.col("total")).alias("proportion")
            )
            p = (
                ggplot(proportions, aes(x=self.var1, y="proportion", fill=self.var2))
                + geom_col(position=position_dodge(), alpha=0.8)
                + labs(x=self.var1, y="Proportion", fill=self.var2)
                + ggtitle(f'Proportion comparison by {self.var1}')
                + theme_bw()
            )
            return p

        else:
            raise ValueError(f"Invalid plot type: {plots}. Use 'bar' or 'dodge'.")
