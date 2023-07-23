from cmath import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from pyrsm.utils import ifelse
from pyrsm.radiant import utils as ru
from typing import Optional, Union
from scipy.stats import chisquare


class goodness:
    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        var: str,
        probs: Optional[tuple[float, ...]] = None,
        figsize: tuple[float, float] = None,
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        self.var = var
        self.figsize = figsize
        self.probs = probs

        self.freq = self.data[self.var].value_counts().to_dict()
        self.nlev = len(self.freq)
        if self.probs is None:
            self.probs = [1 / self.nlev] * self.nlev

        self.observed = pd.DataFrame(
            {k: [v] for k, v in self.freq.items()},
            columns=sorted(self.freq.keys()),
        )
        self.observed["Total"] = self.observed[list(self.observed.columns)].sum(axis=1)
        self.expected = pd.DataFrame(
            {
                sorted(self.freq.keys())[i]: [
                    self.probs[i] * self.observed.at[0, "Total"],
                ]
                for i in range(len(self.freq.keys()))
            },
            columns=sorted(self.freq.keys()),
        )
        self.expected["Total"] = self.expected[list(self.expected.columns)].sum(axis=1)
        self.chisq = pd.DataFrame(
            {
                column: [
                    round(
                        (
                            (self.observed.at[0, column] - self.expected.at[0, column])
                            ** 2
                        )
                        / self.expected.at[0, column],
                        2,
                    ),
                ]
                for column in self.expected.columns.tolist()[:-1]
            },
            columns=self.expected.columns.tolist(),
        )
        self.chisq["Total"] = self.chisq[list(self.chisq.columns)].sum(axis=1)

        self.stdev = pd.DataFrame(
            {
                column: [
                    round(
                        (self.observed.at[0, column] - self.expected.at[0, column])
                        / sqrt(self.expected.at[0, column]).real,
                        2,
                    ),
                ]
                for column in self.expected.columns.tolist()[:-1]
            },
            columns=self.expected.columns.tolist()[:-1],
        )

    def summary(
        self, output: list[str] = ["observed", "expected"], dec: int = 3
    ) -> None:

        pd.set_option("display.max_columns", 20)
        pd.set_option("display.max_rows", 20)

        output = ifelse(isinstance(output, str), [output], output)

        print("Goodness of fit test")
        print(f"Data         : {self.name}")
        if self.var not in self.data.columns:
            raise ValueError(f"{self.var} does not exist in chosen dataset")

        print(f"Variable     : {self.var}")
        if self.nlev != len(self.probs):
            raise ValueError(
                f'Number of elements in "probs" should match the number of levels in {self.var} ({self.nlev})'
            )

        if not 0.999 <= sum(self.probs) <= 1.001:
            raise ValueError("Probabilities do not sum to 1 ({sum(self.probs)})")

        print(f'Probabilities: {" ".join(map(str, self.probs))}')
        print(
            f"Null hyp.    : The distribution of {self.var} is consistent with the specified distribution"
        )
        print(
            f"Alt. hyp.    : The distribution of {self.var} is not consistent with the specified distribution"
        )

        if "observed" in output:
            print("\nObserved:")
            print(self.observed.to_string(index=False))

        if "expected" in output:
            print("\nExpected: total x p")
            print(self.expected.to_string(index=False))

        if "chisq" in output:
            print("\nContribution to chi-squared: (observed - expected) ^ 2 / expected")
            print(self.chisq.to_string(index=False))

        if "dev_std" in output:
            print("\nDeviation standardized: (observed - expected) / sqrt(expected)\n")
            print(self.stdev.to_string(index=False))

        chisq, p_val = chisquare(
            [self.freq[key] for key in sorted(self.freq.keys())],
            [self.expected.at[0, key] for key in sorted(self.freq.keys())],
        )

        if p_val < 0.001:
            p_val = "< .001"
        else:
            p_val = round(p_val, dec)
        print(
            f"\nChi-squared: {round(chisq, dec)} df ({self.nlev - 1}), p.value {p_val}"
        )

    def plot(self, plots: list[str] = [], **kwargs) -> None:
        plots = ifelse(isinstance(plots, str), [plots], plots)

        args = {"rot": False}
        if "observed" in plots:
            tab = self.observed.copy().drop(columns="Total").transpose()
            args["title"] = "Observed frequencies"
            args.update(**kwargs)
            fig = tab.plot.bar(**args, legend=False)
        if "expected" in plots:
            tab = self.expected.copy().drop(columns="Total").transpose()
            args["title"] = "Expected frequencies"
            args.update(**kwargs)
            fig = tab.plot.bar(**args, legend=False)
        if "chisq" in plots:
            tab = self.chisq.drop(columns="Total").transpose()
            args["title"] = "Contribution to chi-squared statistic"
            args.update(**kwargs)
            fig = tab.plot.bar(**args, legend=False)
        if "dev_std" in plots:
            tab = self.stdev.transpose()
            args["title"] = "Deviation standardized"
            args.update(**kwargs, legend=False)
            fig, ax = plt.subplots()
            tab.plot.bar(**args, ax=ax)
            ax.axhline(y=1.96, color="black", linestyle="--")
            ax.axhline(y=1.64, color="black", linestyle="--")
            ax.axhline(y=0, color="black")
            ax.axhline(y=-1.96, color="black", linestyle="--")
            ax.axhline(y=-1.64, color="black", linestyle="--")
            ax.annotate("95%", xy=(0, 2.1), va="bottom", ha="center")
            ax.annotate("90%", xy=(0, 1.4), va="top", ha="center")


if __name__ == "__main__":
    import pyrsm as rsm

    data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
    gf = rsm.basics.goodness(data=data_dct, var="Income", probs=[1 / 2, 1 / 2])
    gf.summary(output=["observed"])
    gf.plot(plots=["observed", "expected", "chisq", "dev_std"])
