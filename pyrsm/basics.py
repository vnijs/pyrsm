import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from pyrsm.logit import sig_stars


class cross_tabs:
    def __init__(self, df, var1, var2):
        self.df = df
        self.var1 = var1
        self.var2 = var2

        self.observed = pd.crosstab(
            df[var1], columns=df[var2], margins=True, margins_name="Total"
        )
        self.chisq = stats.chi2_contingency(
            self.observed.drop(columns="Total").drop("Total", axis=0), correction=False
        )
        expected = pd.DataFrame(self.chisq[3])
        expected["Total"] = expected.sum(axis=1)
        expected = expected.append(expected.sum(), ignore_index=True).set_index(
            self.observed.index
        )
        expected.columns = self.observed.columns
        self.expected = expected
        self.chi_sq = (
            ((self.observed - self.expected) ** 2 / self.expected)
            .drop(columns="Total")
            .drop("Total", axis=0)
        )
        self.perc_row = self.observed.div(self.observed["Total"], axis=0)
        self.perc_col = self.observed.div(self.observed.loc["Total", :], axis=1)
        self.perc = self.observed / self.observed.loc["Total", "Total"]

    def summary(self, output=["observed", "expected"], dec=2):
        prn = f"""
Cross-tabs
Variables: {self.var1}, {self.var2}
Null hyp: there is no association between {self.var1} and {self.var2}
Alt. hyp: there is an association between {self.var1} and {self.var2}
"""
        if "observed" in output:
            prn = (
                prn
                + f"""
Observed:

{self.observed.applymap(lambda x: "{:,}".format(x))}
"""
            )
        if "observed" in output:
            prn = (
                prn
                + f"""
Expected: (row total x column total) / total

{self.expected.round(dec).applymap(lambda x: "{:,}".format(x))}
"""
            )
        if "chisq" in output:
            prn = (
                prn
                + f"""
Contribution to chi-squared: (o - e)^2 / e

{self.chi_sq.round(dec).applymap(lambda x: "{:,}".format(x))}
"""
            )
        if "perc_row" in output:
            prn = (
                prn
                + f"""
Row percentages:

{self.perc_row.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
            )
        if "perc_col" in output:
            prn = (
                prn
                + f"""
Column percentages:

{self.perc_col.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
            )

        if "perc_all" in output:
            prn = (
                prn
                + f"""
Percentages:

{self.perc.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
            )

        prn = (
            prn
            + f"""
Chi-squared: {round(self.chisq[0], dec)} df({round(self.chisq[2], dec)}), p.value {round(self.chisq[1], dec)}
"""
        )
        print(prn)

    # Another instance method
    def plot(self, output="perc_col", **kwargs):
        pdf = getattr(self, output).drop(columns="Total").drop("Total", axis=0)
        fig = pdf.plot.bar(**kwargs)


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
        df = pd.DataFrame({"x": [0, 1, 1, 1, 0], "y": [1, 0, 0, 0, np.NaN]})
        c = correlation(df)
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

    def summary(self, dec=2):
        """
        Print correlations between numeric variables in a Pandas dataframe

        Parameters
        ----------
        dec : int
            Number of decimal places to use in rounding

        Examples
        --------
        df = pd.DataFrame({"x": [0, 1, 1, 1, 0], "y": [1, 0, 0, 0, np.NaN]})
        correlation(df).summary()
        """
        ind = np.triu_indices(self.cr.shape[0])
        cn = self.df.columns[:-1]
        indn = self.df.columns[1:]

        # correlations
        crs = self.cr.round(dec).astype(str)
        crs[ind] = ""
        crs = pd.DataFrame(
            np.delete(np.delete(crs, 0, axis=0), crs.shape[1] - 1, axis=1),
            columns=cn,
            index=indn,
        )

        # pvalues
        cps = self.cp.round(dec).astype(str)
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

        s = "Correlation\n"
        s += "Variables: " + ", ".join(list(self.df.columns)) + "\n"
        s += "Null hyp.: " + f"variables {x} and {y} are not correlated\n"
        s += "Alt. hyp.: " + f"variables {x} and {y} are correlated\n"
        print(s)
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
            Number of observations to use for the scatter plots
        dec : int
            Number of decimal places to use in rounding
        figsize : tuple
            A tuple that determines the figure size. If None, size is
            determined based on the number of numeric variables in the
            data

        Examples
        --------
        df = pd.DataFrame({"x": [0, 1, 1, 1, 0], "y": [1, 0, 0, 0, np.NaN]})
        correlation(df).plot()
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
