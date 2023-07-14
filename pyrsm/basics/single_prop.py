from cmath import sqrt
import pandas as pd
from scipy import stats
from typing import Union


class single_prop:
    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        variable: str,
        level: str,
        alt_hyp: str,
        conf: float,
        comp_value: float,
        test_type: str = "binomial",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name].copy()
        else:
            self.data = data.copy()
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
        return "Plotting not yet implemented"


if __name__ == "__main__":
    import pyrsm as rsm

    consider, consider_description = rsm.load_data(pkg="basics", name="consider")
    sp = single_prop(
        data={"consider": consider},
        variable="consider",
        level="yes",
        alt_hyp="less",
        conf=0.95,
        comp_value=0.1,
        test_type="binomial",
    )
    sp.summary()
    sp.plot()
