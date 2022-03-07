import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from pyrsm.logit import sig_stars
from pyrsm.utils import ifelse


class cross_tabs:
    def __init__(self, df, var1, var2):
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

    def summary(self, output=["observed", "expected"], dec=2):
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

    def plot(self, output="perc_col", **kwargs):
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
    def __init__(self, df):
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

    def summary(self, cutoff=0, dec=2):
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

    def plot(self, nobs=1000, dec=2, figsize=None):
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
        rsm.load_data(pkg="basics", name="salary", dct=globals())
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

            fig, axes = plt.subplots(ncol, ncol, figsize=figsize)

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
