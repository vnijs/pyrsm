from cmath import sqrt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from typing import Union
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats import multitest
import pyrsm.basics.utils as bu
import pyrsm.radiant.utils as ru
from pyrsm.model import sig_stars
from pyrsm.utils import ifelse, check_dataframe


class compare_props:
    """
    Compare proportions across levels of a categorical variable in a Pandas
    or Polars dataframe. See the notebook linked below for a worked example,
    including the web UI:

    https://github.com/vnijs/pyrsm/blob/main/examples/basics-compare-means.ipynb
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        var1: str,
        var2: str,
        lev: str,
        comb: list[tuple[str, str]] = [],
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
        # self.comb = comb
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.alpha = 1 - self.conf
        self.adjust = adjust

        self.data[self.var1] = self.data[self.var1].astype("category")
        self.levels = list(self.data[self.var1].cat.categories)

        if len(comb) == 0:
            self.comb = ru.iterms(self.levels)
        else:
            self.comb = ifelse(isinstance(comb, str), [comb], comb)

        descriptive_stats = []
        for lev in self.levels:
            subset = self.data.loc[
                (self.data[self.var2] == self.lev) & (self.data[self.var1] == lev),
                self.var2,
            ]
            ns = len(subset)
            n_missing = subset.isna().sum()
            n = len(self.data.loc[(self.data[self.var1] == lev), self.var2]) - n_missing
            p = ns / n
            sd = sqrt(p * (1 - p)).real
            se = (sd / sqrt(n)).real
            z_score = stats.norm.ppf(1 - self.alpha / 2)
            me = (z_score * sd / sqrt(n)).real
            descriptive_stats.append([lev, ns, p, n, n_missing, sd, se, me])

        self.descriptive_stats = pd.DataFrame(
            descriptive_stats,
            columns=[
                self.var1,
                self.lev,
                "p",
                "n",
                "n_missing",
                "sd",
                "se",
                "me",
            ],
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

            x = self.data.loc[self.data[self.var1] == v1, self.var2].dropna()
            y = self.data.loc[self.data[self.var1] == v2, self.var2].dropna()

            c1 = np.sum(x == self.lev)
            c2 = np.sum(y == self.lev)
            n1 = len(x)
            n2 = len(y)
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
        self.comp_stats = pd.DataFrame(
            comp_stats,
            columns=[
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

    def summary(self, extra=False, dec: int = 3) -> None:
        comp_stats = self.comp_stats.copy()
        if not extra:
            comp_stats = comp_stats.iloc[:, [0, 1, 2, 3, -1]]

        comp_stats["p.value"] = ifelse(
            comp_stats["p.value"] < 0.001, "< .001", round(comp_stats["p.value"], dec)
        )
        print("Pairwise proportion comparisons")
        print(f"Data      : {self.name}")
        print(f"Variables : {self.var1}, {self.var2}")
        print(f'Level     : "{self.lev}" in {self.var2}')
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.adjust}\n")
        print(self.descriptive_stats.round(dec).to_string(index=False), "\n")
        print(comp_stats.round(dec).to_string(index=False))
        print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    def plot(self, plots: str = "bar") -> None:
        if plots == "bar":
            data = self.data[[self.var1, self.var2]].copy()
            data[self.var2] = ifelse(data[self.var2] == self.lev, 1.0, 0.0)
            sns.barplot(
                data=data,
                y=self.var2,
                x=self.var1,
                # yerr=self.descriptive_stats["se"], weird issue with se's ...
            )
        elif plots == "dodge":
            data = self.data[[self.var1, self.var2]].copy()
            pt = (
                data.groupby(self.var1, observed=False)
                .value_counts(normalize=True)
                .reset_index()
            )
            sns.barplot(
                data=pt,
                y="proportion",
                x=self.var1,
                hue=self.var2,
                dodge=True,
            )
        else:
            print("Invalid plot type")
