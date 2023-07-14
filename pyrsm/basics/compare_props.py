from cmath import sqrt
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union


class compare_props:
    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        grouping_var: str,
        var: str,
        level: str,
        combinations: list[tuple[str, str]],
        alt_hyp: str,
        conf: float,
        multiple_comp_adjustment: str = "none",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"
        self.grouping_var = grouping_var
        self.var = var
        self.level = level
        self.combinations = combinations
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.multiple_comp_adjustment = multiple_comp_adjustment

        # self.p_val = None

        # print("Pairwise proportion comparisons")

        # def calculate(self) -> None:
        combinations_elements = set()
        for combination in self.combinations:
            combinations_elements.add(combination[0])
            combinations_elements.add(combination[1])
        combinations_elements = list(combinations_elements)

        rows1 = []
        for element in combinations_elements:
            subset = self.data[
                (self.data[self.var] == self.level)
                & (self.data[self.grouping_var] == element)
            ][self.var]
            ns = len(subset)
            n_missing = subset.isna().sum()
            n = (
                len(self.data[(self.data[self.grouping_var] == element)][self.var])
                - n_missing
            )
            p = ns / n
            # print(f"ns: {ns}, n: {n}, p: {p}, nmissing: {n_missing}")
            sd = sqrt(p * (1 - p))
            se = sd / sqrt(n)
            z_score = stats.norm.ppf((1 + self.conf) / 2)
            # was printing out imaginary part in som cases
            me = np.real(z_score * sd / sqrt(n))
            row = [element, ns, p, n, n_missing, sd, se, me]
            rows1.append(row)

        self.table1 = pd.DataFrame(
            rows1,
            columns=[
                self.grouping_var,
                self.level,
                "p",
                "n",
                "n_missing",
                "sd",
                "se",
                "me",
            ],
        )

        alt_hyp_sign = " > "
        if self.alt_hyp == "less":
            alt_hyp_sign = " < "
        elif self.alt_hyp == "two-sided":
            alt_hyp_sign = " != "

        rows2 = []
        for v1, v2 in self.combinations:
            null_hyp = v1 + " = " + v2
            alt_hyp = v1 + alt_hyp_sign + v2

            subset1 = self.data[
                (self.data[self.var] == self.level)
                & (self.data[self.grouping_var] == v1)
            ][self.var]

            ns1 = len(subset1)
            n_missing1 = subset1.isna().sum()
            n1 = (
                len(self.data[(self.data[self.grouping_var] == v1)][self.var])
                - n_missing1
            )
            p1 = ns1 / n1

            subset2 = self.data[
                (self.data[self.var] == self.level)
                & (self.data[self.grouping_var] == v2)
            ][self.var]

            ns2 = len(subset2)
            n_missing2 = subset2.isna().sum()
            n2 = (
                len(self.data[self.data[self.grouping_var] == v2][self.var])
                - n_missing2
            )
            p2 = ns2 / n2

            diff = p1 - p2

            observed = pd.crosstab(
                self.data[v1], columns=self.data[v2], margins=True, margins_name="Total"
            )
            chisq, self.p_val, df, _ = stats.chi2_contingency(
                self.observed.drop(columns="Total").drop("Total", axis=0),
                correction=False,
            )
            # chisq, self.p_val, df, _ = stats.chi2_contingency()  # unsure about this

            # print(f"chisq: {chisq}")

            row = [
                null_hyp,
                alt_hyp,
                diff,
                self.p_val,
                chisq,
                df,
                # zero_percent,
                # x_percent,
            ]
            rows2.append(row)

        self.table2 = pd.DataFrame(
            rows2,
            columns=[
                "Null hyp.",
                "Alt. hyp.",
                "diff",
                "p.value",
                "chisq.value",
                "df",
                # "0%",
                # str(self.conf * 100) + "%",
            ],
        )

    def summary(self, dec: int = 3) -> None:
        if hasattr(self.data, "description"):
            data_name = self.data.description.split("\n")[0].split()[1].lower()
        else:
            data_name = "Not available"
        print(f"Data: {data_name}")
        print(f"Variables: {self.grouping_var}, {self.var}")
        print(f"Level: {self.level} in {self.var}")
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.multiple_comp_adjustment}\n")
        print(self.table1.round(dec).to_string(index=False))
        print(self.table2.round(dec).to_string(index=False))
