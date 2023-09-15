import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pyrsm.utils import ifelse
from typing import Union


class cross_tabs:
    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        var1: str,
        var2: str,
    ) -> None:
        """
        Calculate a Chi-square test between two categorical variables contained
        in a Pandas dataframe

        Parameters
        ----------
        data : Pandas dataframe with categorical variables or a
            dictionary with a single dataframe as the value and the
            name of the dataframe as the key

        Returns
        -------
        Cross object with several attributes
        data: Original dataframe
        var1: Name of the first categorical variable
        var2: Name of the second categorical variable
        observed: Dataframe of observed frequencies
        expected: Dataframe of expected frequencies
        expected_low: List with number of cells with expected values < 5
            and the total number of cells
        chisq: Dataframe of chi-square values for each cell
        dev_std: Dataframe of standardized deviations from the expected table
        perc_row: Dataframe of observation percentages conditioned by row
        perc_col: Dataframe of observation percentages conditioned by column
        perc: Dataframe of observation percentages by the total number of observations

        Examples
        --------
        import pyrsm as rsm
        rsm.load_data(pkg="basics", name="newspaper", dct=globals())
        ct = rsm.cross_tabs(newspaper, "Income", "Newspaper")
        ct.expected
        """
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"
        self.var1 = var1
        self.var2 = var2

        self.observed = pd.crosstab(
            self.data[var1], columns=self.data[var2], margins=True, margins_name="Total"
        )
        self.chisq_test = stats.chi2_contingency(
            self.observed.drop(columns="Total").drop("Total", axis=0), correction=False
        )
        expected = pd.DataFrame(self.chisq_test[3])
        self.expected_low = [
            (expected < 5).sum().sum(),
            expected.shape[0] * expected.shape[0],
        ]
        expected["Total"] = expected.sum(axis=1)
        expected = pd.concat(
            [expected, expected.sum().to_frame().T], ignore_index=True, axis=0
        ).set_index(self.observed.index)
        expected.columns = self.observed.columns
        self.expected = expected
        chisq = (self.observed - self.expected) ** 2 / self.expected
        chisq["Total"] = chisq.sum(axis=1)
        chisq.loc["Total", :] = chisq.sum()
        self.chisq = chisq
        self.dev_std = (
            ((self.observed - self.expected) / np.sqrt(self.expected))
            .drop(columns="Total")
            .drop("Total", axis=0)
        )
        self.perc_row = self.observed.div(self.observed["Total"], axis=0)
        self.perc_col = self.observed.div(self.observed.loc["Total", :], axis=1)
        self.perc = self.observed / self.observed.loc["Total", "Total"]

    def summary(
        self, output: list[str] = ["observed", "expected"], dec: int = 2
    ) -> None:
        """
        Print different output tables for a cross_tabs object

        Parameters
        ----------
        output : list of tables to show
            Options include "observed" (observed frequencies),
            "expected" (expected frequencies), "chisq" (chi-square values)
            for each cell, "dev_std" (standardized deviations from expected)
            "perc_row" (percentages conditioned by row), "perc_col"
            (percentages conditioned by column), "perc" (percentages by the
            total number of observations). The default value is ["observed", "expected"]
        dec : int
            Number of decimal places to use in rounding

        Examples
        --------
        import pyrsm as rsm
        rsm.load_data(pkg="basics", name="newspaper", dct=globals())
        ct = rsm.cross_tabs(newspaper)
        ct.summary()
        """

        pd.set_option("display.max_columns", 20)
        pd.set_option("display.max_rows", 20)

        output = ifelse(isinstance(output, str), [output], output)
        prn = f"""
Cross-tabs
Variables: {self.var1}, {self.var2}
Data     : {self.name}
Null hyp : There is no association between {self.var1} and {self.var2}
Alt. hyp : There is an association between {self.var1} and {self.var2}
"""
        if "observed" in output:
            prn += f"""
Observed:

{self.observed.applymap(lambda x: "{:,}".format(x))}
"""
        if "expected" in output:
            prn += f"""
Expected: (row total x column total) / total

{self.expected.round(dec).applymap(lambda x: "{:,}".format(x))}
"""
        if "chisq" in output:
            prn += f"""
Contribution to chi-squared: (o - e)^2 / e

{self.chisq.round(dec).applymap(lambda x: "{:,}".format(x))}
"""

        if "dev_std" in output:
            prn += f"""
Deviation standardized: (o - e) / sqrt(e)

{self.dev_std.round(dec).applymap(lambda x: "{:,}".format(x))}
"""

        if "perc_row" in output:
            prn += f"""
Row percentages:

{self.perc_row.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
        if "perc_col" in output:
            prn += f"""
Column percentages:

{self.perc_col.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
        if "perc" in output:
            prn += f"""
Percentages:

{self.perc.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
        prn += f"""
Chi-squared: {round(self.chisq_test[0], dec)} df({int(self.chisq_test[2])}), p.value {ifelse(self.chisq_test[1] < 0.001, "< .001", round(self.chisq_test[1], dec))}
{100 * round(self.expected_low[0] / self.expected_low[1], dec)}% of cells have expected values below 5
"""
        print(prn)

        pd.reset_option("display.max_columns")
        pd.reset_option("display.max_rows")

    def plot(self, plots: list[str] = "perc_col", **kwargs) -> None:
        """
        Plot of correlations between numeric variables in a Pandas dataframe

        Parameters
        ----------
        plots : list of tables to show
            Options include "observed" (observed frequencies),
            "expected" (expected frequencies), "chisq" (chi-square values)
            for each cell, "dev_std" (standardized deviations from expected)
            "perc_row" (percentages conditioned by row), "perc_col"
            (percentages conditioned by column), "perc" (percentages by the
            total number of observations). The default value is ["observed", "expected"]
        **kwargs : Named arguments to be passed to pandas plotting functions

        Examples
        --------
        import pyrsm as rsm
        rsm.load_data(pkg="basics", name="newspaper", dct=globals())
        ct = rsm.cross_tabs(newspaper, "Income", "Newspaper")
        ct.plot()
        """
        plots = ifelse(isinstance(plots, str), [plots], plots)

        args = {"rot": False}
        if "observed" in plots:
            tab = (
                self.observed.transpose()
                .drop(columns="Total")
                .drop("Total", axis=0)
                .apply(lambda x: x * 100 / sum(x), axis=1)
            )
            args["title"] = "Observed frequencies"
            args.update(**kwargs)
            fig = tab.plot.bar(stacked=True, **args)
        if "expected" in plots:
            tab = (
                self.expected.transpose()
                .drop(columns="Total")
                .drop("Total", axis=0)
                .apply(lambda x: x * 100 / sum(x), axis=1)
            )
            args["title"] = "Expected frequencies"
            args.update(**kwargs)
            fig = tab.plot.bar(stacked=True, **args)
        if "chisq" in plots:
            tab = self.chisq.transpose().drop(columns="Total").drop("Total", axis=0)
            args["title"] = "Contribution to chi-squared statistic"
            args.update(**kwargs)
            fig = tab.plot.bar(**args)
        if "dev_std" in plots:
            tab = self.dev_std.transpose()
            args["title"] = "Deviation standardized"
            args.update(**kwargs)
            fig, ax = plt.subplots()
            tab.plot.bar(**args, ax=ax)
            ax.axhline(y=1.96, color="black", linestyle="--")
            ax.axhline(y=1.64, color="black", linestyle="--")
            ax.axhline(y=0, color="black")
            ax.axhline(y=-1.96, color="black", linestyle="--")
            ax.axhline(y=-1.64, color="black", linestyle="--")
            ax.annotate("95%", xy=(0, 2.1), va="bottom", ha="center")
            ax.annotate("90%", xy=(0, 1.4), va="top", ha="center")
        if "perc_col" in plots:
            tab = self.perc_col.transpose().drop(columns="Total").drop("Total", axis=0)
            args["title"] = "Column percentages"
            args.update(**kwargs)
            fig = tab.plot.bar(**args)
        if "perc_row" in plots:
            tab = self.perc_row.transpose().drop(columns="Total").drop("Total", axis=0)
            args["title"] = "Row percentages"
            args.update(**kwargs)
            fig = tab.plot.bar(**args)
        if "perc" in plots:
            tab = self.perc.transpose().drop(columns="Total").drop("Total", axis=0)
            args["title"] = "Table percentages"
            args.update(**kwargs)
            fig = tab.plot.bar(**args)
