from cmath import sqrt
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union
from statsmodels.stats import multitest
import pyrsm.basics.utils as bu
import pyrsm.radiant.utils as ru


class compare_props:
    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        gvar: str,
        var: str,
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
        self.gvar = gvar
        self.var = var
        self.lev = lev
        self.comb = comb
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.alpha = 1 - self.conf
        self.adjust = adjust

        self.data[self.gvar] = self.data[self.gvar].astype("category")
        self.levels = list(self.data[self.gvar].cat.categories)

        if len(self.comb) == 0:
            self.comb = ru.iterms(self.levels)

        descriptive_stats = []
        for lev in self.levels:
            subset = self.data[
                (self.data[self.var] == self.lev) & (self.data[self.gvar] == lev)
            ][self.var]
            ns = len(subset)
            n_missing = subset.isna().sum()
            n = len(self.data[(self.data[self.gvar] == lev)][self.var]) - n_missing
            p = ns / n
            sd = sqrt(p * (1 - p)).real
            se = (sd / sqrt(n)).real
            z_score = stats.norm.ppf(1 - self.alpha / 2)
            me = (z_score * sd / sqrt(n)).real
            descriptive_stats.append([lev, ns, p, n, n_missing, sd, se, me])

        self.descritive_stats = pd.DataFrame(
            descriptive_stats,
            columns=[
                self.gvar,
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

        # rows2 = []
        # for v1, v2 in self.comb:
        #     null_hyp = v1 + " = " + v2
        #     alt_hyp = v1 + alt_hyp_sign + v2

        #     subset1 = self.data[
        #         (self.data[self.var] == self.lev) & (self.data[self.gvar] == v1)
        #     ][self.var]

        #     ns1 = len(subset1)
        #     n_missing1 = subset1.isna().sum()
        #     n1 = len(self.data[(self.data[self.gvar] == v1)][self.var]) - n_missing1
        #     p1 = ns1 / n1

        #     subset2 = self.data[
        #         (self.data[self.var] == self.lev) & (self.data[self.gvar] == v2)
        #     ][self.var]

        #     ns2 = len(subset2)
        #     n_missing2 = subset2.isna().sum()
        #     n2 = len(self.data[self.data[self.gvar] == v2][self.var]) - n_missing2
        #     p2 = ns2 / n2

        #     diff = p1 - p2

        #     observed = pd.crosstab(
        #         self.data[v1], columns=self.data[v2], margins=True, margins_name="Total"
        #     )
        #     chisq, self.p_val, df, _ = stats.chi2_contingency(
        #         self.observed.drop(columns="Total").drop("Total", axis=0),
        #         correction=False,
        #     )
        #     # chisq, self.p_val, df, _ = stats.chi2_contingency()  # unsure about this

        #     # print(f"chisq: {chisq}")

        #     row = [
        #         null_hyp,
        #         alt_hyp,
        #         diff,
        #         self.p_val,
        #         chisq,
        #         df,
        #         # zero_percent,
        #         # x_percent,
        #     ]
        #     rows2.append(row)

        # self.table2 = pd.DataFrame(
        #     rows2,
        #     columns=[
        #         "Null hyp.",
        #         "Alt. hyp.",
        #         "diff",
        #         "p.value",
        #         "chisq.value",
        #         "df",
        #         # "0%",
        #         # str(self.conf * 100) + "%",
        #     ],
        # )

    def summary(self, extra=False, dec: int = 3) -> None:
        print(f"Pairwise proportion comparisons")
        print(f"Data      : {self.name}")
        print(f"Variables : {self.gvar}, {self.var}")
        print(f'Level     : "{self.lev}" in {self.var}')
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.adjust}\n")
        print(self.descritive_stats.round(dec).to_string(index=False))
        # print(self.table2.round(dec).to_string(index=False))
