from cmath import sqrt
from typing import Union

import numpy as np
import polars as pl
from scipy import stats
from plotnine import (
    aes,
    geom_col,
    ggplot,
    ggtitle,
    labs,
    scale_y_continuous,
    theme_bw,
)

import pyrsm.basics.utils as bu
import pyrsm.basics.display_utils as du
import pyrsm.basics.plotting_utils as pu
from pyrsm.model.model import sig_stars
from pyrsm.utils import check_dataframe, ifelse


class single_prop:
    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        var: str,
        lev: str = None,
        alt_hyp: str = "two-sided",
        conf: float = 0.95,
        comp_value: float = 0.5,
        test_type: str = "binomial",
    ) -> None:
        if comp_value == 0 or comp_value == 1:
            raise Exception("Please choose a comparison value between 0 and 1")

        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"
        self.data = check_dataframe(self.data)
        self.var = var
        self.lev = lev
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.alpha = 1 - self.conf
        self.comp_value = comp_value
        self.test_type = test_type
        self.ns = self.data.filter(pl.col(self.var) == self.lev).height
        self.n_missing = self.data[self.var].null_count()
        self.n = self.data.height - self.n_missing
        self.p = self.ns / self.n
        self.sd = sqrt(self.p * (1 - self.p)).real
        self.se = (self.sd / sqrt(self.n)).real
        self.se_p0 = sqrt(self.comp_value * (1 - self.comp_value) / self.n).real
        self.z_critical = stats.norm.ppf(1 - self.alpha / 2)
        self.z_score = None
        self.me = self.z_critical * self.se
        self.diff = self.p - self.comp_value

        def wilson_ci(zc):
            # Wilson CI used by R's prop.test function
            z_square_n = zc**2 / self.n
            denominator = 1 + z_square_n
            lower_limit = (
                self.p
                + z_square_n / 2
                - zc * np.sqrt(self.se**2 + z_square_n / (4 * self.n))
            ) / denominator
            upper_limit = (
                self.p
                + z_square_n / 2
                + zc * np.sqrt(self.se**2 + z_square_n / (4 * self.n))
            ) / denominator
            return [lower_limit, upper_limit]

        if test_type == "binomial":
            result = stats.binomtest(self.ns, self.n, self.comp_value, self.alt_hyp)
            self.ci = result.proportion_ci(confidence_level=conf)
            self.p_val = result.pvalue
        else:
            self.z_score = (self.p - self.comp_value) / self.se_p0
            p_val = stats.norm.cdf(self.z_score)
            if self.alt_hyp == "two-sided":
                self.p_val = p_val * 2
                self.ci = wilson_ci(self.z_critical)
            elif self.alt_hyp == "less":
                self.p_val = p_val
                self.ci = [0, wilson_ci(stats.norm.ppf(self.conf))[1]]
            else:
                self.p_val = 1 - p_val
                self.ci = [wilson_ci(stats.norm.ppf(self.conf))[0], 1]

    def summary(self, dec=3) -> None:
        display = du.SummaryDisplay(
            header_func=self._summary_header,
            plain_func=lambda extra, d: self._summary_plain(d),
            styled_func=lambda extra, d: self._style_tables(d),
        )
        display.display(extra=False, dec=dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        print(
            f'Single proportion ({ifelse(self.test_type=="binomial", "binomial exact", "z-test")})'
        )
        print(f"Data      : {self.name}")
        print(f"Variable  : {self.var}")
        print(f'Level     : "{self.lev}" in {self.var}')
        print(f"Confidence: {self.conf}")
        print(
            f'Null hyp. : the proportion of "{self.lev}" in {self.var} is equal to {self.comp_value}'
        )

        if self.alt_hyp == "less":
            alt_hyp = "less than"
        elif self.alt_hyp == "two-sided":
            alt_hyp = "not equal to"
        else:
            alt_hyp = "greater than"

        print(
            f'Alt. hyp. : the proportion of "{self.lev}" in {self.var} {alt_hyp} {self.comp_value}\n'
        )

    def _summary_plain(self, dec: int = 3) -> None:
        """Print plain text tables."""
        cl = bu.ci_label(self.alt_hyp, self.conf, dec=dec)

        stats_df = pl.DataFrame({
            "p": [round(self.p, dec)],
            "ns": [self.ns],
            "n": [self.n],
            "n_missing": [self.n_missing],
            "sd": [round(self.sd, dec)],
            "se": [round(self.se, dec)],
            "me": [round(self.me, dec)],
        })

        statistic_name = ifelse(self.test_type == "binomial", "ns", "z.value")
        statistic_val = ifelse(
            self.test_type == "binomial",
            self.ns,
            round(self.z_score, dec) if self.z_score is not None else None
        )
        p_val_str = ifelse(self.p_val < 0.001, "< .001", round(self.p_val, dec))

        test_df = pl.DataFrame({
            "diff": [round(self.diff, dec)],
            statistic_name: [statistic_val],
            "p.value": [p_val_str],
            cl[0]: [round(self.ci[0], dec)],
            cl[1]: [round(self.ci[1], dec)],
            "": [sig_stars([self.p_val])[0]],
        })

        du.print_plain_tables(stats_df, test_df)

    def _style_tables(self, dec: int = 3) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        cl = bu.ci_label(self.alt_hyp, self.conf, dec=dec)

        stats_df = pl.DataFrame({
            "p": [self.p],
            "ns": [self.ns],
            "n": [self.n],
            "n_missing": [self.n_missing],
            "sd": [self.sd],
            "se": [self.se],
            "me": [self.me],
        })

        statistic_name = ifelse(self.test_type == "binomial", "ns", "z.value")
        statistic_val = ifelse(
            self.test_type == "binomial",
            self.ns,
            self.z_score if self.z_score is not None else None
        )

        test_df = pl.DataFrame({
            "diff": [self.diff],
            statistic_name: [statistic_val],
            "p.value": [du.format_pval(self.p_val, dec)],
            cl[0]: [self.ci[0]],
            cl[1]: [self.ci[1]],
            "": [sig_stars([self.p_val])[0]],
        })

        gt1 = du.style_table(
            stats_df,
            title="Descriptive Statistics",
            subtitle=f'Level: "{self.lev}" in {self.var}',
            number_cols=["p", "sd", "se", "me"],
            integer_cols=["ns", "n", "n_missing"],
            dec=dec,
        )

        number_cols = ["diff", cl[0], cl[1]]
        if statistic_name == "z.value":
            number_cols.append("z.value")

        gt2 = du.style_table(
            test_df,
            title="Hypothesis Test",
            subtitle=f"Comparison value: {self.comp_value}",
            number_cols=number_cols,
            integer_cols=["ns"] if statistic_name == "ns" else None,
            dec=dec,
        )

        display(gt1)
        display(gt2)

    def plot(self, plots: str = "bar"):
        """
        Plot proportions for single proportion test.

        Parameters
        ----------
        plots : str
            Plot type (default "bar")

        Returns
        -------
        plotnine.ggplot
            A plotnine ggplot object
        """
        if plots == "bar":
            # Get value counts as proportions using polars
            counts = self.data.group_by(self.var).len().sort(self.var)
            total = counts["len"].sum()
            proportions = counts.with_columns((pl.col("len") / total).alias("proportion"))

            p = (
                ggplot(proportions, aes(x=self.var, y="proportion"))
                + geom_col(fill=pu.PlotConfig.FILL, alpha=0.7)
                + scale_y_continuous(labels=lambda x: [f"{v:.0%}" for v in x])
                + labs(x=self.var, y="")
                + ggtitle(f'Single proportion: "{self.lev}" in {self.var}')
                + theme_bw()
            )
            return p


if __name__ == "__main__":
    import pyrsm as rsm

    consider, consider_description = rsm.load_data(pkg="basics", name="consider")
    sp = single_prop(
        data={"consider": consider},
        var="consider",
        lev="yes",
        alt_hyp="less",
        conf=0.95,
        comp_value=0.1,
        test_type="binomial",
    )
    sp.summary()
    sp.plot()
