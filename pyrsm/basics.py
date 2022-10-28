from cmath import sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from .logit import sig_stars
from .utils import ifelse
from typing import Any, Optional
from scipy.stats import chisquare


class cross_tabs:
    def __init__(self, df: pd.DataFrame, var1: str, var2: str) -> None:
        """
        Calculate a Chi-square test between two categorical variables contained
        in a Pandas dataframe

        Parameters
        ----------
        df : Pandas dataframe with numeric variables

        Returns
        -------
        Cross object with several attributes
        df: Original dataframe
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
        self.df = df
        self.var1 = var1
        self.var2 = var2

        self.observed = pd.crosstab(
            df[var1], columns=df[var2], margins=True, margins_name="Total"
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

        output = ifelse(type(output) is list, output, [output])
        prn = f"""
Cross-tabs
Variables: {self.var1}, {self.var2}
Null hyp: there is no association between {self.var1} and {self.var2}
Alt. hyp: there is an association between {self.var1} and {self.var2}
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
        if "perc_all" in output:
            prn += f"""
Percentages:

{self.perc.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
        prn += f"""
Chi-squared: {round(self.chisq_test[0], dec)} df({int(self.chisq_test[2])}), p.value {ifelse(self.chisq_test[1] < 0.001, "< .001", round(self.chisq_test[1], dec))}
{100 * round(self.expected_low[0] / self.expected_low[1], dec)}% of cells have expected values below 5
"""
        print(prn)

    def plot(self, output: list[str] = "perc_col", **kwargs) -> None:
        """
        Plot of correlations between numeric variables in a Pandas dataframe

        Parameters
        ----------
        output : list of tables to show
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
        output = ifelse(type(output) is list, output, [output])

        args = {"rot": False}
        if "observed" in output:
            df = (
                self.observed.transpose()
                .drop(columns="Total")
                .drop("Total", axis=0)
                .apply(lambda x: x * 100 / sum(x), axis=1)
            )
            args["title"] = "Observed frequencies"
            args.update(**kwargs)
            fig = df.plot.bar(stacked=True, **args)
        if "expected" in output:
            df = (
                self.expected.transpose()
                .drop(columns="Total")
                .drop("Total", axis=0)
                .apply(lambda x: x * 100 / sum(x), axis=1)
            )
            args["title"] = "Expected frequencies"
            args.update(**kwargs)
            fig = df.plot.bar(stacked=True, **args)
        if "chisq" in output:
            df = self.chisq.transpose().drop(columns="Total").drop("Total", axis=0)
            args["title"] = "Contribution to chi-squared statistic"
            args.update(**kwargs)
            fig = df.plot.bar(**args)
        if "dev_std" in output:
            df = self.dev_std.transpose()
            args["title"] = "Deviation standardized"
            args.update(**kwargs)
            fig, ax = plt.subplots()
            df.plot.bar(**args, ax=ax)
            ax.axhline(y=1.96, color="black", linestyle="--")
            ax.axhline(y=1.64, color="black", linestyle="--")
            ax.axhline(y=-1.96, color="black", linestyle="--")
            ax.axhline(y=-1.64, color="black", linestyle="--")
            ax.annotate("95%", xy=(0, 2.1), va="bottom", ha="center")
            ax.annotate("90%", xy=(0, 1.4), va="top", ha="center")
        if "perc_col" in output:
            df = self.perc_col.transpose().drop(columns="Total").drop("Total", axis=0)
            args["title"] = "Column percentages"
            args.update(**kwargs)
            fig = df.plot.bar(**args)
        if "perc_row" in output:
            df = self.perc_row.transpose().drop(columns="Total").drop("Total", axis=0)
            args["title"] = "Row percentages"
            args.update(**kwargs)
            fig = df.plot.bar(**args)
        if "perc" in output:
            df = self.perc.transpose().drop(columns="Total").drop("Total", axis=0)
            args["title"] = "Table percentages"
            args.update(**kwargs)
            fig = df.plot.bar(**args)


class correlation:
    def __init__(
        self, df: pd.DataFrame, figsize: Optional[tuple[float, float]] = None
    ) -> None:
        """
        Calculate correlations between numeric variables in a Pandas dataframe

        Parameters
        ----------
        df : Pandas dataframe with numeric variables

        Returns
        -------
        Correlation object with two key attributes
        cr: Correlation matrix
        cp: p.value matrix

        Examples
        --------
        import pandas as pd
        import pyrsm as rsm
        rsm.load_data(pkg="basics", name="salary", dct=globals())
        cr = rsm.correlation(salary[["salary", "yrs.since.phd", "yrs.service"]])
        c.cr
        """
        df = df.copy()
        isNum = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col].dtype)
        ]
        df = df[isNum]

        ncol = df.shape[1]
        cr = np.zeros([ncol, ncol])
        cp = cr.copy()
        for i in range(ncol - 1):
            for j in range(i + 1, ncol):
                cdf = df.iloc[:, [i, j]]
                # pairwise deletion
                cdf = cdf[~np.any(np.isnan(cdf), axis=1)]
                c = stats.pearsonr(cdf.iloc[:, 0], cdf.iloc[:, 1])
                cr[j, i] = c[0]
                cp[j, i] = c[1]

        self.df = df
        self.cr = cr
        self.cp = cp
        self.figsize = figsize

    def summary(self, cutoff: float = 0, dec: int = 2) -> None:
        """
        Print correlations between numeric variables in a Pandas dataframe

        Parameters
        ----------
        cutoff : float
            Only show correlations larger than a threshold in absolute value
        dec : int
            Number of decimal places to use in rounding

        Examples
        --------
        import pandas as pd
        import pyrsm as rsm
        rsm.load_data(pkg="basics", name="salary", dct=globals())
        cr = rsm.correlation(salary[["salary", "yrs.since.phd", "yrs.service"]])
        cr.summary()
        """
        ind = np.triu_indices(self.cr.shape[0])
        cn = self.df.columns[:-1]
        indn = self.df.columns[1:]

        # correlations
        crs = self.cr.round(dec).astype(str)
        if cutoff > 0:
            crs[np.abs(self.cr) < cutoff] = ""
        crs[ind] = ""
        crs = pd.DataFrame(
            np.delete(np.delete(crs, 0, axis=0), crs.shape[1] - 1, axis=1),
            columns=cn,
            index=indn,
        )

        # pvalues
        cps = self.cp.round(dec).astype(str)
        if cutoff > 0:
            cps[np.abs(self.cr) < cutoff] = ""
        cps[ind] = ""
        cps = pd.DataFrame(
            np.delete(np.delete(cps, 0, axis=0), cps.shape[1] - 1, axis=1),
            columns=cn,
            index=indn,
        )

        cn = self.df.columns
        if len(cn) > 2:
            x = "x"
            y = "y"
        else:
            x = cn[0]
            y = cn[1]

        prn = "Correlation\n"
        prn += "Variables: " + ", ".join(list(self.df.columns)) + "\n"
        prn += f"Null hyp.: variables {x} and {y} are not correlated\n"
        prn += f"Alt. hyp.: variables {x} and {y} are correlated\n"
        print(prn)
        print("Correlation matrix:")
        print(crs)
        print("\np.values:")
        print(cps)

    def plot(self, nobs: int = 1000, dec: int = 2, figsize: tuple[float, float] = None):
        """
        Plot of correlations between numeric variables in a Pandas dataframe

        Parameters
        ----------
        nobs : int
            Number of observations to use for the scatter plots. The default
            value is 1,000. To use all observations in the plots, use nobs=-1
        dec : int
            Number of decimal places to use in rounding
        figsize : tuple
            A tuple that determines the figure size. If None, size is
            determined based on the number of numeric variables in the
            data

        Examples
        --------
        import pandas as pd
        import pyrsm as rsm
        rsm.(pkg="basics", name="salary", dct=globals())
        cr = rsm.correlation(salary[["salary", "yrs.since.phd", "yrs.service"]])
        cr.plot(figsize=(7, 3))
        """

        def cor_label(label, longest, ax_sub):
            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)
            # set font size to avoid exceeding boundaries
            font = (80 * self.fig_size) / (len(longest) * len(self.df.columns))
            ax_sub.text(
                0.5,
                0.5,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font,
            )

        def cor_text(r, p, ax_sub, dec=2):
            if np.isnan(p):
                p = 1

            p = round(p, dec)
            rt = round(r, dec)
            p_star = sig_stars([p])[0]

            font = (50 * self.fig_size) / (len(str(rt)) * len(self.df.columns))
            font_star = (12 * self.fig_size) / len(self.df.columns)

            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)
            ax_sub.text(
                0.5,
                0.5,
                rt,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font * abs(r),
            )
            ax_sub.text(
                0.8,
                0.8,
                p_star,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font_star,
                color="blue",
            )

        def cor_plot(x_data, y_data, ax_sub, s_size):
            sns.regplot(
                x=x_data,
                y=y_data,
                ax=ax_sub,
                scatter_kws={"alpha": 0.3, "color": "black", "s": s_size},
                line_kws={"color": "blue"},
            )

            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)

        def cor_mat(df, cmat, pmat, dec=2, nobs=1000, figsize=None):

            cn = df.columns
            ncol = len(cn)
            longest = max(cn, key=len)
            s_size = 50 / len(self.df.columns)

            if figsize is None:
                figsize = (max(5, cmat.shape[0]), max(cmat.shape[0], 5))

            self.fig_size = min(figsize[0], figsize[1])
            s_size = (5 * self.fig_size) / len(self.df.columns)

            _, axes = plt.subplots(ncol, ncol, figsize=figsize)

            if nobs < df.shape[0] and nobs != np.Inf and nobs != -1:
                df = df.copy().sample(nobs)

            for i in range(ncol):
                for j in range(ncol):
                    if i == j:
                        cor_label(cn[i], longest, axes[i, j])
                    elif i > j:
                        cor_plot(df[cn[i]], df[cn[j]], axes[i, j], s_size)
                    else:
                        cor_text(cmat[j, i], pmat[j, i], axes[i, j], dec=2)

            plt.subplots_adjust(wspace=0.04, hspace=0.04)
            plt.show()

        cor_mat(self.df, self.cr, self.cp, dec=dec, nobs=nobs, figsize=figsize)


class prob_calc:
    # Probability calculator
    def __init__(self, distribution: str, params: dict) -> None:
        self.distribution = distribution
        self.__dict__.update(params)
        print("Probability calculator")

    def calculate(self):
        print(f"Distribution: {self.distribution}")

        def calc_f_dist(
            dfn: int, dfd: int, lb: float = 0, ub: float = 0.95, decimals: int = 3
        ) -> tuple[float, float]:
            print(f"Df 1:\t{dfn}")
            print(f"Df 2:\t{dfd}")

            print(f"Mean:\t{round(stats.f.mean(dfn, dfd, loc=lb), decimals)}")
            print(f"Variance:\t{round(stats.f.var(dfn, dfd, loc=lb), decimals)}")
            print(f"Lower bound:\t{lb}")
            print(f"Upper bound:\t{ub}\n")

            if lb == 0:
                critical_f = round(stats.f.ppf(q=ub, dfn=dfn, dfd=dfd), decimals)

                _num_decimal_places_in_ub = len(str(ub).split(".")[-1])

                print(f"P(X < {critical_f}) = {ub}")
                print(
                    f"P(X > {critical_f}) = {round(1 - ub, _num_decimal_places_in_ub)}"
                )
                return (0, critical_f)

            critical_f_lower = round(stats.f.ppf(q=lb, dfn=dfn, dfd=dfd), decimals)

            _num_decimal_places_in_lb = len(str(lb).split(".")[-1])

            print(f"P(X < {critical_f_lower}) = {lb}")
            print(
                f"P(X > {critical_f_lower}) = {round(1 - lb, _num_decimal_places_in_lb)}"
            )
            ########################################################################################
            critical_f_upper = round(stats.f.ppf(q=ub, dfn=dfn, dfd=dfd), decimals)

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
            print(f"Df:\t{df}")
            print(f"Mean:\t{round(stats.t.mean(df), decimals)}")
            print(f"St. dev:\t{round(stats.t.std(df), decimals)}")
            print(f"Lower bound:\t{lb}")
            print(f"Upper bound:\t{ub}")
            print()

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
                f"1 - P({critical_t_lower} < X < {critical_t_upper} = {round(1 - (ub - lb), _num_decimal_places)}"
            )

            return (critical_t_lower, critical_t_upper)

        if self.distribution == "F":
            lb = self.lb if "lb" in self.__dict__ else 0
            ub = self.ub if "ub" in self.__dict__ else 0.95
            dfn = self.dfn
            dfd = self.dfd
            decimals = self.decimals if "decimals" in self.__dict__ else 3
            calc_f_dist(dfn, dfd, lb, ub, decimals)

        elif self.distribution == "t":
            lb = self.lb if "lb" in self.__dict__ else 0
            ub = self.ub if "ub" in self.__dict__ else 0.95
            df = self.df
            decimals = self.decimals if "decimals" in self.__dict__ else 3
            calc_t_dist(df, lb, ub, decimals)

    def plot(self):
        def plot_f_dist(
            dfn: int, dfd: int, lb: float = 0, ub: float = 0.95, decimals: int = 3
        ):
            x = np.linspace(stats.f.ppf(0, dfn, dfd), stats.f.ppf(0.99, dfn, dfd), 200)

            plt.grid()
            pdf = stats.f.pdf(x, dfn, dfd)

            plt.plot(x, pdf, "black", lw=1, alpha=0.6, label="f pdf")

            if lb == 0:
                critical_f = round(stats.f.ppf(q=ub, dfn=dfn, dfd=dfd), decimals)
                plt.fill_between(x, pdf, where=(x < critical_f), color="slateblue")
                plt.fill_between(x, pdf, where=(x > critical_f), color="salmon")
            else:
                critical_f_lower = round(stats.f.ppf(q=lb, dfn=dfn, dfd=dfd), decimals)
                critical_f_upper = round(stats.f.ppf(q=ub, dfn=dfn, dfd=dfd), decimals)

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

            plt.grid()
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
            dfn = self.dfn
            dfd = self.dfd
            decimals = self.decimals if "decimals" in self.__dict__ else 3
            plot_f_dist(dfn, dfd, lb, ub, decimals)

        elif self.distribution == "t":
            lb = self.lb if "lb" in self.__dict__ else 0
            ub = self.ub if "ub" in self.__dict__ else 0.95
            df = self.df
            decimals = self.decimals if "decimals" in self.__dict__ else 3
            plot_t_dist(df, lb, ub, decimals)


class single_mean:
    def __init__(
        self,
        data: pd.DataFrame,
        variable: str,
        alt_hypo: str,
        conf: float,
        comparison_value: float,
    ):
        self.data = data
        self.variable = variable
        self.alt_hypo = alt_hypo
        self.conf = conf
        self.comparison_value = comparison_value
        self.t_val = None
        self.p_val = None

        print("Single mean test")

    def calculate(self) -> None:
        result = stats.ttest_1samp(
            a=self.data[self.variable],
            popmean=self.comparison_value,
            nan_policy="omit",
            alternative=self.alt_hypo,
        )

        self.t_val, self.p_val = result.statistic, result.pvalue

        self.mean = np.nanmean(self.data[self.variable])
        self.n = len(self.data[self.variable])
        self.n_missing = self.data[self.variable].isna().sum()

        self.sd = self.data[self.variable].std()
        self.se = self.data[self.variable].sem()
        z_score = stats.norm.ppf((1 + self.conf) / 2)

        self.me = z_score * self.sd / sqrt(self.n)
        self.diff = self.mean - self.comparison_value
        self.df = self.n - 1
        self.x_percent = self.mean - stats.t.ppf(self.conf, self.df) * self.se
        self.hundred_percent = self.mean - stats.t.ppf(0, self.df) * self.se

    def summary(self, dec=3) -> None:
        if self.t_val == None:
            self.calculate()
        data_name = ""
        if hasattr(self.data, "description"):
            data_name = self.data.description.split("\n")[0].split()[1].lower()
        if len(data_name) > 0:
            print(f"Data: {data_name}")
        print(f"Variable: {self.variable}")
        print(f"Confidence: {self.conf}")

        print(f"Null hyp.: the mean of {self.variable} = {self.comparison_value}")

        alt_hypo = ">"
        if self.alt_hypo == "lesser":
            alt_hypo = "<"
        elif self.alt_hypo == "two-sided":
            alt_hypo = "!="

        print(
            f"Alt. hyp.: the mean of {self.variable} is {alt_hypo} {self.comparison_value}\n"
        )

        row1 = [[self.mean, self.n, self.n_missing, self.sd, self.se, self.me]]
        row2 = [
            [
                self.diff,
                self.se,
                self.t_val,
                ifelse(self.p_val < 0.001, "< .001", self.p_val),
                self.df,
                self.x_percent,
                self.hundred_percent,
            ]
        ]

        col_names1 = ["mean", "n", "n_missing", "sd", "se", "me"]
        col_names2 = [
            "diff",
            "se",
            "t.value",
            "p.value",
            "df",
            str(int((1 - self.conf) * 100)) + "%",
            "100%",
        ]

        table1 = pd.DataFrame(row1, columns=col_names1).round(dec)
        table2 = pd.DataFrame(row2, columns=col_names2).round(dec)

        print(table1.to_string(index=False))
        print(table2.to_string(index=False))

    def plot(self, types: list[str], figsize: tuple[float, float] = (10, 10)) -> None:
        numplots = 2
        which_plot = ""
        if isinstance(types, str):
            numplots = 1
            which_plot = types
        elif isinstance(types, list):
            assert len(types) == 2
            assert "hist" in types and "sim" in types
        else:
            raise TypeError("`types` should be of list data type")

        _, axes = plt.subplots(numplots, figsize=figsize)

        if numplots == 1:
            if which_plot == "hist":
                self.data[self.variable].plot.hist(
                    ax=axes, title=self.variable, color="slateblue"
                )
                plt.sca(axes)
                plt.vlines(
                    x=(self.comparison_value, self.x_percent, self.mean),
                    ymin=axes.get_ylim()[0],
                    ymax=axes.get_ylim()[1],
                    colors=("r", "k", "k"),
                    linestyles=("solid", "dashed", "solid"),
                )
            else:
                # TODO: ask Prof. Nijs about this
                pass
        else:
            self.data[self.variable].plot.hist(
                ax=axes[0], title=self.variable, color="slateblue"
            )
            plt.vlines(
                x=[self.comparison_value, self.me, self.mean],
                ymin=0,
                ymax=axes.get_ylim(),
                colors=["r", "k", "k"],
                linestyles=["solid", "dashed", "dashed"],
            )
            # simulate plot here


class compare_means:
    def __init__(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str,
        combinations: list[tuple[str, str]],
        alt_hypo: str,
        conf: float,
        sample_type: str = "independent",
        multiple_comp_adjustment: str = "none",
        test_type: str = "t-test",
    ) -> None:
        self.data = data
        self.var1 = var1
        self.var2 = var2
        self.combinations = combinations
        self.alt_hypo = alt_hypo
        self.conf = conf
        self.sample_type = sample_type
        self.multiple_comp_adjustment = multiple_comp_adjustment
        self.test_type = test_type

        self.t_val = None
        self.p_val = None

        print(f"Pairwise mean comparisons {self.test_type}")

    def welch_dof(self, v1: str, v2: str) -> float:
        x = self.data[self.data[self.var1] == v1][self.var2]
        y = self.data[self.data[self.var1] == v2][self.var2]
        dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
            (x.var() / x.size) ** 2 / (x.size - 1)
            + (y.var() / y.size) ** 2 / (y.size - 1)
        )

        return dof

    def calculate(self) -> None:
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
            z_score = stats.norm.ppf((1 + self.conf) / 2)
            # was printing out imaginary part in som cases
            me = np.real(z_score * sd / sqrt(n))
            row = [element, mean, n, n_missing, sd, se, me]
            rows1.append(row)

        self.table1 = pd.DataFrame(
            rows1, columns=["rank", "mean", "n", "n_missing", "sd", "se", "me"]
        )

        alt_hypo_sign = " > "
        if self.alt_hypo == "less":
            alt_hypo_sign = " < "
        elif self.alt_hypo == "two-sided":
            alt_hypo_sign = " != "

        rows2 = []
        for v1, v2 in self.combinations:
            null_hypo = v1 + " = " + v2
            alt_hypo = v1 + alt_hypo_sign + v2
            diff = np.nanmean(
                self.data[self.data[self.var1] == v1][self.var2]
            ) - np.nanmean(self.data[self.data[self.var1] == v2][self.var2])

            result = stats.ttest_ind(
                self.data[self.data[self.var1] == v1][self.var2],
                self.data[self.data[self.var1] == v2][self.var2],
                equal_var=False,
                nan_policy="omit",
                alternative=self.alt_hypo,
            )

            self.t_val, self.p_val = result.statistic, result.pvalue
            se = self.data[self.data[self.var1] == v2][self.var2].sem()
            df = self.welch_dof(v1, v2)

            """
            Not entirely sure how to calculate these
            """
            # zero_percent = mean - stats.t.ppf(1, df) * se
            # x_percent = mean - stats.t.ppf(self.conf, df) * se

            row = [
                null_hypo,
                alt_hypo,
                diff,
                self.t_val,
                ifelse(self.p_val < 0.001, "< .001", self.p_val),
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
                "t.value",
                "p.value",
                "df",
                # "0%",
                # str(self.conf * 100) + "%",
            ],
        )

    def summary(self, dec=3) -> None:
        if self.t_val == None:
            self.calculate()
        data_name = ""
        if hasattr(self.data, "description"):
            data_name = self.data.description.split("\n")[0].split()[1].lower()
        if len(data_name) > 0:
            print(f"Data: {data_name}")
        print(f"Variables: {self.var1}, {self.var2}")
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.multiple_comp_adjustment}")

        print(self.table1.round(dec).to_string(index=False))
        print(self.table2.round(dec).to_string(index=False))


class single_prop:
    def __init__(
        self,
        data: pd.DataFrame,
        variable: str,
        level: Any,
        alt_hypo: str,
        conf: float,
        comparison_value: float,
        test_type: str = "binomial",
    ) -> None:
        self.data = data
        self.variable = variable
        self.level = level
        self.alt_hypo = alt_hypo
        self.conf = conf
        self.comparison_value = comparison_value
        self.test_type = test_type

        self.p_val = None

        print(f"Single proportion ({self.test_type})")

    def calculate(self) -> None:
        self.ns = len(self.data[self.data[self.variable] == self.level])
        self.n_missing = self.data[self.variable].isna().sum()
        self.n = len(self.data) - self.n_missing
        self.p = self.ns / self.n
        self.sd = sqrt(self.p * (1 - self.p))
        self.se = self.sd / sqrt(self.n)

        z_score = stats.norm.ppf((1 + self.conf) / 2)

        self.me = z_score * self.se

        self.diff = self.p - self.comparison_value

        results = stats.binomtest(self.ns, self.n, self.comparison_value, self.alt_hypo)

        self.p_val = results.pvalue
        proportion_ci = results.proportion_ci(self.conf)
        self.zero_percent = proportion_ci.low
        self.x_percent = proportion_ci.high

    def summary(self) -> None:
        if self.p_val == None:
            self.calculate()
        data_name = ""
        if hasattr(self.data, "description"):
            data_name = self.data.description.split("\n")[0].split()[1].lower()
        if len(data_name) > 0:
            print(f"Data: {data_name}")

        print(f"Variable: {self.variable}")
        print(f"Level: {self.level} in {self.variable}")
        print(f"Confidence: {self.conf}")
        print(
            f"Null hyp.: the proportion of {self.level} in {self.variable} = {self.comparison_value}"
        )

        alt_hypo_sign = ">"
        if self.alt_hypo == "less":
            alt_hypo_sign = "<"
        elif self.alt_hypo == "two-sided":
            alt_hypo_sign = "!="

        print(
            f"Alt. hyp.: the proportion of {self.level} in {self.variable} {alt_hypo_sign} {self.comparison_value}"
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


class compare_props:
    def __init__(
        self,
        data: pd.DataFrame,
        grouping_var: str,
        var: str,
        level: str,
        combinations: list[tuple[str, str]],
        alt_hypo: str,
        conf: float,
        multiple_comp_adjustment: str = "none",
    ) -> None:
        self.data = data
        self.grouping_var = grouping_var
        self.var = var
        self.level = level
        self.combinations = combinations
        self.alt_hypo = alt_hypo
        self.conf = conf
        self.multiple_comp_adjustment = multiple_comp_adjustment

        self.p_val = None

        print("Pairwise proportion comparisons")

    def calculate(self) -> None:
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
            print(f"ns: {ns}, n: {n}, p: {p}, nmissing: {n_missing}")
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

        alt_hypo_sign = " > "
        if self.alt_hypo == "less":
            alt_hypo_sign = " < "
        elif self.alt_hypo == "two-sided":
            alt_hypo_sign = " != "

        rows2 = []
        for v1, v2 in self.combinations:
            null_hypo = v1 + " = " + v2
            alt_hypo = v1 + alt_hypo_sign + v2

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

            chisq, self.p_val, df, _ = stats.chi2_contingency()  # unsure about this
            # print(f"chisq: {chisq}")

            row = [
                null_hypo,
                alt_hypo,
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
        if self.p_val == None:
            self.calculate()
        data_name = ""
        if hasattr(self.data, "description"):
            data_name = self.data.description.split("\n")[0].split()[1].lower()
        if len(data_name) > 0:
            print(f"Data: {data_name}")

        print(f"Variables: {self.grouping_var}, {self.var}")
        print(f"Level: {self.level} in {self.var}")
        print(f"Confidence: {self.conf}")
        print(f"Adjustment: {self.multiple_comp_adjustment}")

        print()

        print(self.table1.round(dec).to_string(index=False))
        print(self.table2.round(dec).to_string(index=False))


class central_limit_theorem:
    def __init__(
        self,
        dist: str,
        sample_size: int,
        num_samples: int,
        num_bins: int,
        figsize: Optional[tuple[float, float]] = None,
        **params: float,
    ) -> None:
        self.dist = dist
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.num_bins = max(1, min(num_bins, 50))
        self.figsize = figsize
        self.params = params

    def simulate(self) -> None:
        if self.dist == "normal":
            self._simulate_normal()
        elif self.dist == "binomial":
            self._simulate_binomial()
        elif self.dist == "uniform":
            self._simulate_uniform()
        elif self.dist == "exponential":
            self._simulate_exponential()
        else:
            print("Invalid distribution")

    def _simulate_normal(self) -> None:
        mean = self.params["mean"]
        sd = self.params["sd"]
        samples = [
            np.random.normal(loc=mean, scale=sd, size=self.sample_size)
            for _ in range(self.num_samples)
        ]

        self.plot_distribution(samples)

    def _simulate_binomial(self) -> None:
        size = self.params["size"]
        prob = self.params["prob"]

        samples = [
            np.random.binomial(n=size, p=prob, size=self.sample_size)
            for _ in range(self.num_samples)
        ]

        self.plot_distribution(samples)

    def _simulate_uniform(self) -> None:
        minimum = self.params["min"]
        maximum = self.params["max"]

        samples = [
            np.random.uniform(low=minimum, high=maximum, size=self.sample_size)
            for _ in range(self.num_samples)
        ]

        self.plot_distribution(samples)

    def _simulate_exponential(self) -> None:
        rate = self.params["rate"]

        samples = [
            np.random.exponential(scale=rate, size=self.sample_size)
            for _ in range(self.num_samples)
        ]

        self.plot_distribution(samples)

    def plot_distribution(self, samples: list[np.ndarray]) -> None:
        sample_means = [np.mean(sample) for sample in samples]

        _, axes = plt.subplots(2, 2, figsize=self.figsize)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        plt.axes(axes[0][0])
        self._plot_distribution(x=samples[0], x_label="Histogram of sample #1").plot()

        plt.axes(axes[0][1])
        self._plot_distribution(
            x=samples[-1], x_label="Histogram of sample #" + str(self.num_samples)
        ).plot()

        plt.axes(axes[1][0])
        self._plot_distribution(
            x=sample_means, x_label="Histogram of sample means"
        ).plot()

        plt.axes(axes[1][1])
        axes[1][1].set_ylabel("y")
        self._plot_distribution(
            x=sample_means, x_label="Density of sample means", density_plot=True
        ).plot()

        plt.show()

    def _plot_distribution(
        self, x: list[np.ndarray], x_label: str, density_plot: bool = False
    ) -> matplotlib.axes.Axes:
        stat = "count"
        data = {x_label: x}

        data = pd.DataFrame(data)
        if density_plot:
            return sns.kdeplot(data=data, x=x_label, fill=True)

        return sns.histplot(
            data=data,
            x=x_label,
            stat=stat,
            bins=self.num_bins,
        )


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
        data_name = ""
        if hasattr(self.data, "description"):
            data_name = self.data.description.split("\n")[0].split()[1].lower()
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

        plt.show()
