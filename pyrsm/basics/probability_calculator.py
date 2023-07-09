import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class prob_calc:
    # Probability calculator
    def __init__(self, distribution: str, params: dict) -> None:
        self.distribution = distribution
        self.__dict__.update(params)
        print("Probability calculator")

    def summary(self):
        print(f"Distribution: {self.distribution}")

        def calc_f_dist(
            df1: int, df2: int, lb: float = 0, ub: float = 0.95, decimals: int = 3
        ) -> tuple[float, float]:
            print(f"Df 1        : {df1}")
            print(f"Df 2        : {df2}")
            print(f"Mean        : {round(stats.f.mean(df1, df2, loc=lb), decimals)}")
            print(f"Variance    : {round(stats.f.var(df1, df2, loc=lb), decimals)}")
            print(f"Lower bound : {lb}")
            print(f"Upper bound : {ub}\n")

            if lb == 0:
                critical_f = round(stats.f.ppf(q=ub, dfn=df1, dfd=df2), decimals)

                _num_decimal_places_in_ub = len(str(ub).split(".")[-1])

                print(f"P(X < {critical_f}) = {ub}")
                print(
                    f"P(X > {critical_f}) = {round(1 - ub, _num_decimal_places_in_ub)}"
                )
                return (0, critical_f)

            critical_f_lower = round(stats.f.ppf(q=lb, dfn=df1, dfd=df2), decimals)

            _num_decimal_places_in_lb = len(str(lb).split(".")[-1])

            print(f"P(X < {critical_f_lower}) = {lb}")
            print(
                f"P(X > {critical_f_lower}) = {round(1 - lb, _num_decimal_places_in_lb)}"
            )
            ########################################################################################
            critical_f_upper = round(stats.f.ppf(q=ub, dfn=df1, dfd=df2), decimals)

            _num_decimal_places_in_ub = len(str(ub).split(".")[-1])

            print(f"P(X < {critical_f_upper}) = {ub}")
            print(
                f"P(X > {critical_f_upper}) = {round(1 - ub, _num_decimal_places_in_ub)}"
            )
            ########################################################################################
            _num_decimal_places = max(
                len(str(ub).split(".")[-1]), len(str(lb).split(".")[-1])
            )

            print(
                f"P({critical_f_lower} < X < {critical_f_upper}) = {round((ub - lb), _num_decimal_places)}"
            )
            print(
                f"1 - P({critical_f_lower} < X < {critical_f_upper} = {round(1 - (ub - lb), _num_decimal_places)}"
            )

            return (critical_f_lower, critical_f_upper)

        def calc_t_dist(
            df: int, lb: float = 0, ub: float = 0.95, decimals: int = 3
        ) -> tuple[float, float]:
            print(f"Df          : {df}")
            print(f"Mean        : {round(stats.t.mean(df), decimals)}")
            print(f"St. dev     : {round(stats.t.std(df), decimals)}")
            print(f"Lower bound : {lb}")
            print(f"Upper bound : {ub}\n")

            if lb == 0:
                critical_t = round(stats.t.ppf(q=ub, df=df), decimals)

                _num_decimal_places_in_ub = len(str(ub).split(".")[-1])

                print(f"P(X < {critical_t}) = {ub}")
                print(
                    f"P(X > {critical_t}) = {round(1 - ub, _num_decimal_places_in_ub)}"
                )
                return (0, critical_t)

            critical_t_lower = round(stats.t.ppf(q=lb, df=df), decimals)

            _num_decimal_places_in_lb = len(str(lb).split(".")[-1])

            print(f"P(X < {critical_t_lower}) = {lb}")
            print(
                f"P(X > {critical_t_lower}) = {round(1 - lb, _num_decimal_places_in_lb)}"
            )
            ########################################################################################
            critical_t_upper = round(stats.t.ppf(q=ub, df=df), decimals)

            _num_decimal_places_in_ub = len(str(ub).split(".")[-1])

            print(f"P(X < {critical_t_upper}) = {ub}")
            print(
                f"P(X > {critical_t_upper}) = {round(1 - ub, _num_decimal_places_in_ub)}"
            )
            ########################################################################################
            _num_decimal_places = max(
                len(str(ub).split(".")[-1]), len(str(lb).split(".")[-1])
            )

            print(
                f"P({critical_t_lower} < X < {critical_t_upper}) = {round((ub - lb), _num_decimal_places)}"
            )
            print(
                f"1 - P({critical_t_lower} < X < {critical_t_upper}) = {round(1 - (ub - lb), _num_decimal_places)}"
            )

            return (critical_t_lower, critical_t_upper)

        if self.distribution == "F":
            lb = self.lb if "lb" in self.__dict__ else 0
            ub = self.ub if "ub" in self.__dict__ else 0.95
            df1 = self.df1
            df2 = self.df2
            decimals = self.decimals if "decimals" in self.__dict__ else 3
            calc_f_dist(df1, df2, lb, ub, decimals)

        elif self.distribution == "t":
            lb = self.lb if "lb" in self.__dict__ else 0
            ub = self.ub if "ub" in self.__dict__ else 0.95
            df = self.df
            decimals = self.decimals if "decimals" in self.__dict__ else 3
            calc_t_dist(df, lb, ub, decimals)

    def plot(self):
        def plot_f_dist(
            df1: int, df2: int, lb: float = 0, ub: float = 0.95, decimals: int = 3
        ):
            x = np.linspace(stats.f.ppf(0, df1, df2), stats.f.ppf(0.99, df1, df2), 200)
            pdf = stats.f.pdf(x, df1, df2)
            plt.plot(x, pdf, "black", lw=1, alpha=0.6, label="f pdf")

            if lb == 0:
                critical_f = round(stats.f.ppf(q=ub, dfn=df1, dfd=df2), decimals)
                plt.fill_between(x, pdf, where=(x < critical_f), color="slateblue")
                plt.fill_between(x, pdf, where=(x > critical_f), color="salmon")
            else:
                critical_f_lower = round(stats.f.ppf(q=lb, dfn=df1, dfd=df2), decimals)
                critical_f_upper = round(stats.f.ppf(q=ub, dfn=df1, dfd=df2), decimals)

                plt.fill_between(
                    x,
                    pdf,
                    where=((x > critical_f_upper) | (x < critical_f_lower)),
                    color="slateblue",
                )
                plt.fill_between(
                    x,
                    pdf,
                    where=((x > critical_f_upper) | (x < critical_f_lower)),
                    color="salmon",
                )

        def plot_t_dist(
            df: int, lb: float = 0.025, ub: float = 0.975, decimals: int = 3
        ):
            x = np.linspace(stats.t.ppf(0.01, df), stats.t.ppf(0.99, df), 200)
            pdf = stats.t.pdf(x, df)
            plt.plot(x, pdf, "black", lw=1, alpha=0.6, label="t pdf")

            if lb == 0:
                critical_t = round(stats.t.ppf(q=ub, df=df), decimals)
                plt.fill_between(x, pdf, where=(x < critical_t), color="slateblue")
                plt.fill_between(x, pdf, where=(x > critical_t), color="salmon")
            else:
                critical_t_lower = round(stats.t.ppf(q=lb, df=df), decimals)
                critical_t_upper = round(stats.t.ppf(q=ub, df=df), decimals)

                plt.fill_between(
                    x,
                    pdf,
                    where=((x < critical_t_upper) | (x > critical_t_lower)),
                    color="slateblue",
                )
                plt.fill_between(
                    x,
                    pdf,
                    where=((x > critical_t_upper) | (x < critical_t_lower)),
                    color="salmon",
                )

        if self.distribution == "F":
            lb = self.lb if "lb" in self.__dict__ else 0
            ub = self.ub if "ub" in self.__dict__ else 0.95
            df1 = self.df1
            df2 = self.df2
            decimals = self.decimals if "decimals" in self.__dict__ else 3
            plot_f_dist(df1, df2, lb, ub, decimals)

        elif self.distribution == "t":
            lb = self.lb if "lb" in self.__dict__ else 0
            ub = self.ub if "ub" in self.__dict__ else 0.95
            df = self.df
            decimals = self.decimals if "decimals" in self.__dict__ else 3
            plot_t_dist(df, lb, ub, decimals)
