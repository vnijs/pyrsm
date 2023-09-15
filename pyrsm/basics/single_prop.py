from cmath import sqrt
import pandas as pd
import numpy as np
from scipy import stats
from pyrsm.model import sig_stars
from pyrsm.utils import ifelse
import pyrsm.basics.utils as bu
from typing import Union


class single_prop:
    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        var: str,
        lev: str = None,
        alt_hyp: str = "two-sided",
        conf: float = 0.95,
        comp_value: float = 0.5,
        test_type: str = "binomial",
    ) -> None:
        if comp_value == 0 or comp_value == 1:
            raise Exception("Please choose a comparison value between 0 and 1")

        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name].copy()
        else:
            self.data = data.copy()
            self.name = "Not provided"
        self.var = var
        self.lev = lev
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.alpha = 1 - self.conf
        self.comp_value = comp_value
        self.test_type = test_type
        self.ns = len(self.data[self.data[self.var] == self.lev])
        self.n_missing = self.data[self.var].isna().sum()
        self.n = len(self.data) - self.n_missing
        self.p = self.ns / self.n
        self.sd = sqrt(self.p * (1 - self.p)).real
        self.se = (self.sd / sqrt(self.n)).real
        self.se_p0 = sqrt(self.comp_value * (1 - self.comp_value) / self.n).real
        self.z_critical = stats.norm.ppf(1 - self.alpha / 2)
        self.z_score = None
        self.me = self.z_critical * self.se
        self.diff = self.p - self.comp_value

        def wilson_ci(zc):
            # Wilson CI used by R's prop.test function
            z_square_n = zc**2 / self.n
            denominator = 1 + z_square_n
            lower_limit = (
                self.p
                + z_square_n / 2
                - zc * np.sqrt(self.se**2 + z_square_n / (4 * self.n))
            ) / denominator
            upper_limit = (
                self.p
                + z_square_n / 2
                + zc * np.sqrt(self.se**2 + z_square_n / (4 * self.n))
            ) / denominator
            return [lower_limit, upper_limit]

        if test_type == "binomial":
            result = stats.binomtest(self.ns, self.n, self.comp_value, self.alt_hyp)
            self.ci = result.proportion_ci(confidence_level=conf)
            self.p_val = result.pvalue
        else:
            self.z_score = (self.p - self.comp_value) / self.se_p0
            # self.z_score = (self.p - self.comp_value) / self.se_p
            p_val = stats.norm.cdf(self.z_score)
            if self.alt_hyp == "two-sided":
                self.p_val = p_val * 2
                # traditional CI calculation
                # me = self.z_critical * self.se
                # self.ci = [self.p - me, self.p + me]
                self.ci = wilson_ci(self.z_critical)
            elif self.alt_hyp == "less":
                self.p_val = p_val
                # traditional CI calculation
                # z_critical = stats.norm.ppf(self.conf)
                # me = z_critical * self.se
                # self.ci = [0, self.p + me]
                self.ci = [0, wilson_ci(stats.norm.ppf(self.conf))[1]]
            else:
                self.p_val = 1 - p_val
                # traditional CI calculation
                # z_critical = stats.norm.ppf(self.conf)
                # me = z_critical * self.se
                # self.ci = [self.p - me, 1]
                self.ci = [wilson_ci(stats.norm.ppf(self.conf))[0], 1]

            # can't replicate R's z.value with statsmodels
            # var_p0 = self.comp_value * (1 - self.comp_value)
            # self.z_score, self.p_val = proportion.proportions_ztest(
            #     self.ns, self.n, self.comp_value, prop_var=var_p0
            # )

    def summary(self, dec=3) -> None:
        print(
            f'Single proportion ({ifelse(self.test_type=="binomial", "binomial exact", "z-test")})'
        )
        print(f"Data      : {self.name}")
        print(f"Variable  : {self.var}")
        print(f'Level     : "{self.lev}" in {self.var}')
        print(f"Confidence: {self.conf}")
        print(
            f'Null hyp. : the proportion of "{self.lev}" in {self.var} is equal to {self.comp_value}'
        )

        if self.alt_hyp == "less":
            alt_hyp = "less than"
        elif self.alt_hyp == "two-sided":
            alt_hyp = "not equal to"
        else:
            alt_hyp = "greater than"

        cl = bu.ci_label(self.alt_hyp, self.conf, dec=dec)

        print(
            f'Alt. hyp. : the proportion of "{self.lev}" in {self.var} {alt_hyp} {self.comp_value}\n'
        )

        stats_df = pd.DataFrame(
            {
                "p": [self.p],
                "ns": [self.ns],
                "n": [self.n],
                "n_missing": [self.n_missing],
                "sd": [self.sd],
                "se": [self.se],
                "me": [self.me],
            }
        ).round(dec)

        statistic = [
            ifelse(self.test_type == "binomial", "ns", "z.value"),
            ifelse(self.test_type == "binomial", [self.ns], [self.z_score]),
        ]
        test_df = pd.DataFrame(
            {
                "diff": [self.diff],
                statistic[0]: statistic[1],
                "p.value": [ifelse(self.p_val < 0.001, "< .001", self.p_val)],
                cl[0]: [self.ci[0]],
                cl[1]: [self.ci[1]],
                "": [sig_stars([self.p_val])[0]],
            }
        ).round(dec)

        print(stats_df.to_string(index=False), "\n")
        print(test_df.to_string(index=False))
        print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    def plot(self, plots: str = "bar") -> None:
        if plots == "bar":
            proportion_data = self.data[self.var].value_counts(normalize=True)
            ax = proportion_data.plot(kind="bar", alpha=0.5)
            ax.set_ylabel("")
            ax.set_title(f'Single proportion: "{self.lev}" in {self.var}')
            ax.yaxis.set_major_formatter(lambda x, _: "{:.0%}".format(x))


if __name__ == "__main__":
    import pyrsm as rsm

    consider, consider_description = rsm.load_data(pkg="basics", name="consider")
    sp = single_prop(
        data={"consider": consider},
        var="consider",
        lev="yes",
        alt_hyp="less",
        conf=0.95,
        comp_value=0.1,
        test_type="binomial",
    )
    sp.summary()
    sp.plot()
