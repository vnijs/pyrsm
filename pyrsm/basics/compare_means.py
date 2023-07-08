from cmath import sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from ..model import sig_stars
from ..utils import ifelse
from typing import Any, Optional
from scipy.stats import chisquare
from statsmodels.stats import multitest


class compare_means:
    def __init__(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        combinations: list[tuple[str, str]] = None,
        alt_hyp: str = "two-sided",
        conf: float = 0.95,
        sample_type: str = "independent",
        adjust: str = None,
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"
        self.var1 = var1
        self.var2 = var2
        self.combinations = combinations
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.sample_type = sample_type
        self.adjust = adjust
        self.test_type = "t-test"

        def welch_dof(v1: str, v2: str) -> float:
            # stats.ttest_ind uses Welch's t-test when equal_var=False
            # but does not return the degrees of freedom
            x = self.data[self.data[self.var1] == v1][self.var2]
            y = self.data[self.data[self.var1] == v2][self.var2]
            dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
                (x.var() / x.size) ** 2 / (x.size - 1)
                + (y.var() / y.size) ** 2 / (y.size - 1)
            )

            return dof

        combinations_elements = set()
        for combination in self.combinations:
            combinations_elements.add(combination[0])
            combinations_elements.add(combination[1])
        combinations_elements = list(combinations_elements)

        rows1 = []
        mean = 0
        for element in combinations_elements:
            subset = self.data[self.data[self.var1] == element][self.var2]
            row = []
            mean = np.nanmean(subset)
            n_missing = subset.isna().sum()
            n = len(subset) - n_missing
            sd = subset.std()
            se = subset.sem()
            tscore = stats.t.ppf((1 + self.conf) / 2, n - 1)
            me = (tscore * se).real
            row = [element, mean, n, n_missing, sd, se, me]
            rows1.append(row)

        self.table1 = pd.DataFrame(
            rows1, columns=[self.var1, "mean", "n", "n_missing", "sd", "se", "me"]
        )

        if self.alt_hyp == "less":
            alt_hyp_sign = " < "
        elif self.alt_hyp == "two-sided":
            alt_hyp_sign = " != "
        else:
            alt_hyp_sign = " > "

        rows2 = []
        for v1, v2 in self.combinations:
            null_hyp = v1 + " = " + v2
            alt_hyp = v1 + alt_hyp_sign + v2
            diff = np.nanmean(
                self.data[self.data[self.var1] == v1][self.var2]
            ) - np.nanmean(self.data[self.data[self.var1] == v2][self.var2])

            if self.sample_type == "independent":
                result = stats.ttest_ind(
                    self.data[self.data[self.var1] == v1][self.var2],
                    self.data[self.data[self.var1] == v2][self.var2],
                    equal_var=False,
                    nan_policy="omit",
                    alternative=self.alt_hyp,
                )
            else:
                result = stats.ttest_rel(
                    self.data[self.data[self.var1] == v1][self.var2],
                    self.data[self.data[self.var1] == v2][self.var2],
                    nan_policy="omit",
                    alternative=self.alt_hyp,
                )

            t_val, p_val = result.statistic, result.pvalue
            se = diff / t_val
            data = welch_dof(v1, v2)

            if self.alt_hyp == "two-sided":
                tscore = stats.t.ppf((1 + self.conf) / 2, data)
            else:
                tscore = stats.t.ppf(self.conf, data)
            me = (tscore * se).real

            if self.alt_hyp == "less":
                ci = [-np.inf, diff + me]
            elif self.alt_hyp == "two-sided":
                ci = [diff - me, diff + me]
            else:
                ci = [diff - me, np.inf]

            rows2.append(
                [
                    null_hyp,
                    alt_hyp,
                    diff,
                    ifelse(p_val < 0.001, "< .001", p_val),
                    se,
                    t_val,
                    data,
                    ci[0],
                    ci[1],
                    sig_stars([p_val])[0],
                ]
            )

        cl = ci_label(self.alt_hyp, self.conf)
        self.table2 = pd.DataFrame(
            rows2,
            columns=[
                "Null hyp.",
                "Alt. hyp.",
                "diff",
                "p.value",
                "se",
                "t.value",
                "data",
                cl[0],
                cl[1],
                "",
            ],
        )

        if self.adjust is not None:
            self.table2["p.value"] = multitest.multipletests(
                self.table2["p.value"], method=self.adjust
            )[1]

    def summary(self, dec=3) -> None:
        print(f"Pairwise mean comparisons ({self.test_type})")
        print(f"Data      : {self.name}")
        print(f"Variables : {self.var1}, {self.var2}")
        print(f"Samples   : {self.sample_type}")
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.adjust}")

        print(self.table1.round(dec).to_string(index=False))
        print(self.table2.round(dec).to_string(index=False))


class single_prop:
    def __init__(
        self,
        data: pd.DataFrame,
        variable: str,
        level: Any,
        alt_hyp: str,
        conf: float,
        comp_value: float,
        test_type: str = "binomial",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"
        self.variable = variable
        self.level = level
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.comp_value = comp_value
        self.test_type = test_type

        self.ns = len(self.data[self.data[self.variable] == self.level])
        self.n_missing = self.data[self.variable].isna().sum()
        self.n = len(self.data) - self.n_missing
        self.p = self.ns / self.n
        self.sd = sqrt(self.p * (1 - self.p)).real
        self.se = (self.sd / sqrt(self.n)).real

        z_score = stats.norm.ppf((1 + self.conf) / 2).real

        self.me = z_score * self.se

        self.diff = self.p - self.comp_value

        results = stats.binomtest(self.ns, self.n, self.comp_value, self.alt_hyp)

        self.p_val = results.pvalue
        proportion_ci = results.proportion_ci(self.conf)
        self.zero_percent = proportion_ci.low
        self.x_percent = proportion_ci.high

    def summary(self) -> None:
        print(f"Single proportion ({self.test_type})")
        print(f"Data      : {self.name}")
        print(f"Variable  : {self.variable}")
        print(f"Level     : {self.level} in {self.variable}")
        print(f"Confidence: {self.conf}")
        print(
            f"Null hyp.: the proportion of {self.level} in {self.variable} = {self.comp_value}"
        )

        alt_hyp_sign = ">"
        if self.alt_hyp == "less":
            alt_hyp_sign = "<"
        elif self.alt_hyp == "two-sided":
            alt_hyp_sign = "!="

        print(
            f"Alt. hyp.: the proportion of {self.level} in {self.variable} {alt_hyp_sign} {self.comp_value}"
        )

        row1 = [[self.p, self.ns, self.n, self.n_missing, self.sd, self.se, self.me]]
        table1 = pd.DataFrame(
            row1, columns=["p", "ns", "n", "n_missing", "sd", "se", "me"]
        )

        row2 = [[self.diff, self.ns, self.p_val, self.zero_percent, self.x_percent]]
        table2 = pd.DataFrame(
            row2, columns=["diff", "ns", "p_val", "0%", str(self.conf * 100) + "%"]
        )

        print()
        print(table1.to_string(index=False))
        print(table2.to_string(index=False))

    def plot(self) -> None:
        # TODO
        pass
