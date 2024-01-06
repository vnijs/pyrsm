import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from pyrsm.model import sig_stars
from pyrsm.utils import ifelse, check_dataframe
import pyrsm.basics.utils as bu
from typing import Union


class single_mean:
    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        var: str,
        alt_hyp: str = "two-sided",
        conf: float = 0.95,
        comp_value: float = 0,
    ):
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name].copy()
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)
        self.var = var
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.comp_value = comp_value

        result = stats.ttest_1samp(
            a=self.data[self.var],
            popmean=self.comp_value,
            nan_policy="omit",
            alternative=self.alt_hyp,
        )

        self.t_val, self.p_val = result.statistic, result.pvalue
        self.ci = result.confidence_interval(confidence_level=conf)

        self.mean = np.nanmean(self.data[self.var])
        self.n = len(self.data[self.var])
        self.n_missing = self.data[self.var].isna().sum()

        self.sd = self.data[self.var].std()
        self.se = self.data[self.var].sem()
        tscore = stats.t.ppf((1 + self.conf) / 2, self.n - 1)

        self.me = (tscore * self.se).real
        self.diff = self.mean - self.comp_value
        self.df = self.n - 1

    def summary(self, dec=3) -> None:
        print("Single mean test")
        print(f"Data      : {self.name}")
        print(f"Variables : {self.var}")
        print(f"Confidence: {self.conf}")
        print(f"Comparison: {self.comp_value}\n")
        print(f"Null hyp. : the mean of {self.var} is equal to {self.comp_value}")

        if self.alt_hyp == "less":
            alt_hyp = "less than"
        elif self.alt_hyp == "two-sided":
            alt_hyp = "not equal to"
        else:
            alt_hyp = "greater than"

        cl = bu.ci_label(self.alt_hyp, self.conf, dec=dec)

        print(f"Alt. hyp. : the mean of {self.var} is {alt_hyp} {self.comp_value}\n")

        row1 = [[self.mean, self.n, self.n_missing, self.sd, self.se, self.me]]
        row2 = [
            [
                self.diff,
                self.se,
                self.t_val,
                ifelse(self.p_val < 0.001, "< .001", self.p_val),
                self.df,
                self.ci[0],
                self.ci[1],
                sig_stars([self.p_val])[0],
            ]
        ]

        col_names1 = ["mean", "n", "n_missing", "sd", "se", "me"]
        col_names2 = ["diff", "se", "t.value", "p.value", "df", cl[0], cl[1], ""]

        table1 = pd.DataFrame(row1, columns=col_names1).round(dec)
        table2 = pd.DataFrame(row2, columns=col_names2).round(dec)

        print(table1.to_string(index=False))
        print(table2.to_string(index=False))
        print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    def plot(self, plots: str = "hist") -> None:
        if plots == "hist":
            fig = self.data[self.var].plot.hist(title=self.var, color="slateblue")
            plt.vlines(
                x=(self.comp_value, self.ci[0], self.mean, self.ci[1]),
                ymin=fig.get_ylim()[0],
                ymax=fig.get_ylim()[1],
                colors=("r", "k", "k", "k"),
                linestyles=("solid", "dashed", "solid", "dashed"),
            )
        elif plots == "sim":
            print("Plot type not available yet")
            # self.data[self.var].plot.hist(title=self.var, color="slateblue")
            # plt.vlines(
            #     x=[self.comp_value, self.me, self.mean],
            #     colors=["r", "k", "k"],
            #     linestyles=["solid", "dashed", "dashed"],
            # )
        else:
            print("Invalid plot type")
