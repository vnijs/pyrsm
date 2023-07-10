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


class goodness_of_fit:
    def __init__(
        self,
        data: pd.DataFrame,
        variable: str,
        figsize: tuple[float, float] = None,
        probabilities: Optional[tuple[float, ...]] = None,
        params: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.data = data
        self.variable = variable
        self.figsize = figsize
        self.probabilities = probabilities
        if params is not None:
            params = map(str.lower, params)
            self.params = tuple(params)
        else:
            self.params = params

        self._observed_frequencies = self.data[self.variable].value_counts().to_dict()
        if self.params is not None:
            self._observed_df = pd.DataFrame(
                {
                    key: [
                        item,
                    ]
                    for key, item in self._observed_frequencies.items()
                },
                columns=sorted(self._observed_frequencies.keys()),
            )
            self._observed_df["Total"] = self._observed_df[
                list(self._observed_df.columns)
            ].sum(axis=1)

            self._expected_df = pd.DataFrame(
                {
                    sorted(self._observed_frequencies.keys())[i]: [
                        self.probabilities[i] * self._observed_df.at[0, "Total"],
                    ]
                    for i in range(len(self._observed_frequencies.keys()))
                },
                columns=sorted(self._observed_frequencies.keys()),
            )
            self._expected_df["Total"] = self._expected_df[
                list(self._expected_df.columns)
            ].sum(axis=1)

            self._chisquared_df = pd.DataFrame(
                {
                    column: [
                        round(
                            (
                                (
                                    self._observed_df.at[0, column]
                                    - self._expected_df.at[0, column]
                                )
                                ** 2
                            )
                            / self._expected_df.at[0, column],
                            2,
                        ),
                    ]
                    for column in self._expected_df.columns.tolist()[:-1]
                },
                columns=self._expected_df.columns.tolist(),
            )
            self._chisquared_df["Total"] = self._chisquared_df[
                list(self._chisquared_df.columns)
            ].sum(axis=1)

            self._stdev_df = pd.DataFrame(
                {
                    column: [
                        round(
                            (
                                self._observed_df.at[0, column]
                                - self._expected_df.at[0, column]
                            )
                            / sqrt(self._expected_df.at[0, column]).real,
                            2,
                        ),
                    ]
                    for column in self._expected_df.columns.tolist()[:-1]
                },
                columns=self._expected_df.columns.tolist()[:-1],
            )

    def summary(self) -> None:
        print("Goodness of fit test")
        if hasattr(self.data, "description"):
            data_name = self.data.description.split("\n")[0].split()[1].lower()
        else:
            data_name = "Not available"

        print(f"Data: {data_name}")
        if self.variable not in self.data.columns:
            print(f"{self.variable} does not exist in chosen dataset")
            return

        print(f"Variable: {self.variable}")
        num_levels = self.data[self.variable].nunique()
        if self.probabilities is None:
            self.probabilities = [1 / num_levels] * num_levels

        if num_levels != len(self.probabilities):
            print(
                f'Number of elements in "probabilities" should match the number of levels in {self.variable} ({num_levels})'
            )
            return

        prob_sum = sum(self.probabilities)
        if prob_sum != 1:
            print(f"Probabilities do not sum to 1 ({prob_sum})")
            print(
                f"Use fractions if appropriate. Variable {self.variable} has {num_levels} unique values"
            )
            return

        print(f'Specified: {" ".join(map(str, self.probabilities))}')
        print(
            f"Null hyp.: The distribution of {self.variable} is consistent with the specified distribution"
        )
        print(
            f"Alt. hyp.: The distribution of {self.variable} is not consistent with the specified distribution"
        )

        if self.params is not None:
            if "observed" in self.params:
                print("Observed:")
                print(self._observed_df.to_string(index=False))
                print()

            if "expected" in self.params:
                print("Expected: total x p")
                print(self._expected_df.to_string(index=False))
                print()

            if "chi-squared" in self.params:
                print(
                    "Contribution to chi-squared: (observed - expected) ^ 2 / expected"
                )
                print(self._chisquared_df.to_string(index=False))
                print()

            if "deviation std" in self.params:
                print("Deviation standardized: (observed - expected) / sqrt(expected)")
                print()
                print(self._stdev_df.to_string(index=False))
                print()

        chisq, p_val = chisquare(
            [
                self._observed_frequencies[key]
                for key in sorted(self._observed_frequencies.keys())
            ],
            [
                self._expected_df.at[0, key]
                for key in sorted(self._observed_frequencies.keys())
            ],
        )
        chisq = round(chisq, 3)

        if p_val < 0.001:
            p_val = "< .001"
        print(f"Chi-squared: {chisq} df ({num_levels - 1}), p.value {p_val}")

    def plot(self) -> None:
        _, axes = plt.subplots(2, 2, figsize=self.figsize)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        if "observed" in self.params:
            plt.axes(axes[0][0])
            observed_frequency_percentages_df = pd.DataFrame(
                {
                    "levels": self._observed_df.columns.tolist()[:-1],
                    "percentages": [
                        (
                            self._observed_df.at[0, level]
                            / self._observed_df.at[0, "Total"]
                        )
                        * 100
                        for level in self._observed_df.columns.tolist()[:-1]
                    ],
                }
            )
            sns.barplot(
                data=observed_frequency_percentages_df, x="levels", y="percentages"
            ).plot()

        if "expected" in self.params:
            plt.axes(axes[0][1])
            expected_frequency_percentages_df = pd.DataFrame(
                {
                    "levels": self._expected_df.columns.tolist()[:-1],
                    "percentages": [
                        (
                            self._expected_df.at[0, level]
                            / self._expected_df.at[0, "Total"]
                        )
                        * 100
                        for level in self._expected_df.columns.tolist()[:-1]
                    ],
                }
            )
            sns.barplot(
                data=expected_frequency_percentages_df, x="levels", y="percentages"
            ).plot()

        if "chi-squared" in self.params:
            plt.axes(axes[1][0])
            chisquared_contribution_df = pd.DataFrame(
                {
                    "levels": self._chisquared_df.columns.tolist()[:-1],
                    "contribution": [
                        self._chisquared_df.at[0, level]
                        for level in self._chisquared_df.columns.tolist()[:-1]
                    ],
                }
            )
            sns.barplot(
                data=chisquared_contribution_df, x="levels", y="contribution"
            ).plot()

        if "deviation std" in self.params:
            plt.axes(axes[1][1])
            standardized_deviation_df = pd.DataFrame(
                {
                    "levels": self._stdev_df.columns.tolist(),
                    "stdev": [
                        self._stdev_df.at[0, level]
                        for level in self._stdev_df.columns.tolist()
                    ],
                }
            )

            barplot = sns.barplot(data=standardized_deviation_df, x="levels", y="stdev")

            z_95, z_neg_95 = 1.96, -1.96
            z_90, z_neg_90 = 1.64, -1.64

            barplot.axhline(y=z_95, color="k", linestyle="dashed", linewidth=1)
            plt.annotate(
                "95%",
                xy=(0, z_95),
                xytext=(0, z_95 + 0.1),
                color="black",
                fontsize=7,
            )

            barplot.axhline(y=z_neg_95, color="k", linestyle="dashed", linewidth=1)
            plt.annotate(
                "95%",
                xy=(1, z_neg_95),
                xytext=(1, z_neg_95 - 0.35),
                color="black",
                fontsize=7,
            )

            barplot.axhline(y=z_90, color="k", linestyle="dashed", linewidth=1)
            barplot.axhline(y=z_neg_90, color="k", linestyle="dashed", linewidth=1)

            barplot.plot()

        # plt.show()
