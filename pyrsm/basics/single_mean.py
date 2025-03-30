from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

import pyrsm.basics.utils as bu
from pyrsm.model.model import sig_stars
from pyrsm.utils import check_dataframe, ifelse


class single_mean:
    """
    A class to perform single-mean hypothesis testing

    Attributes
    ----------
    data : pd.DataFrame | pl.DataFrame
        The input data for the hypothesis test as a Pandas or Polars DataFrame. If a dictionary is provided, the key should be the name of the dataframe.
    var : str
        The variable/column name to test.
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
    mean : float
        The mean of the variable.
    n : int
        The number of observations.
    n_missing : int
        The number of missing observations.
    sd : float
        The standard deviation of the variable.
    se : float
        The standard error of the variable.
    me : float
        The margin of error.
    diff : float
        The difference between the mean and the comparison value.
    df : int
        The degrees of freedom.

    Methods
    -------
    __init__(data, var, alt_hyp='two-sided', conf=0.95, comp_value=0)
        Initializes the single_mean class with the provided data and parameters.
    summary(dec=3)
        Prints a summary of the hypothesis test.
    plot()
        Plots the results of the hypothesis test.
    """
    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        var: str,
        alt_hyp: str = "two-sided",
        conf: float = 0.95,
        comp_value: float = 0,
    ):
        """
        Constructs all the necessary attributes for the single_mean object.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame]
            The input data for the hypothesis test as a Pandas or Polars DataFrame.
        var : str
            The variable/column name to test.
        alt_hyp : str, optional
            The alternative hypothesis ('two-sided', 'greater', 'less') (default is 'two-sided').
        conf : float, optional
            The confidence level for the test (default is 0.95).
        comp_value : float, optional
            The comparison value for the test (default is 0).
        """
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

    def summary(self, dec: int = 3) -> None:
        """
        Prints a summary of the hypothesis test.

        Parameters
        ----------
        dec : int, optional
            The number of decimal places to display (default is 3).
        """
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

    def plot(self, plots: Literal["hist", "sim"] = "hist") -> None:
        """
        Plots the results of the hypothesis test. If the 'hist' is selected a histogram
        of the numeric variable will be shown. The solid black line in the histogram shows 
        the sample mean. The dashed black lines show the confidence interval around the 
        sample mean. The solid red line shows the comparison value (i.e., the value under 
        the null-hypothesis). If the red line does not fall within the confidence interval 
        we can reject the null-hypothesis in favor of the alternative at the specified 
        confidence level (e.g., 0.95).

        Parameters
        ----------
        plots : str
            The type of plot to generate (default is 'hist').
 
        """
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
