import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import seaborn as sns
from pyrsm.model.model import sig_stars
from pyrsm.utils import check_dataframe
from typing import Optional, Union


class correlation:
    """
    Calculate correlations between numeric variables in a Pandas dataframe

    Parameters
    ----------
    data : Pandas dataframe with numeric variables

    Returns
    -------
    Correlation object with two key attributes
    cr: Correlation matrix
    cp: p.value matrix
    cv: Covariance matrix

    Examples
    --------
    import pandas as pd
    import pyrsm as rsm
    salary, salary_description = rsm.load_data(pkg="basics", name="salary")
    cr = rsm.correlation(salary[["salary", "yrs.since.phd", "yrs.service"]])
    cr.cr
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        vars: Optional[list[str]] = [],
        method: str = "pearson",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name].copy()
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)
        self.vars = vars
        if len(self.vars) == 0:
            self.vars = [
                col
                for col in self.data.columns
                if pd.api.types.is_numeric_dtype(self.data[col].dtype)
            ]
        self.data = self.data[self.vars].copy()
        self.method = method

        ncol = self.data.shape[1]
        cr = np.zeros([ncol, ncol])
        cp = cr.copy()
        cv = cr.copy()
        for i in range(ncol - 1):
            for j in range(i + 1, ncol):
                cdf = self.data.iloc[:, [i, j]]
                # pairwise deletion
                cdf = cdf[~np.any(np.isnan(cdf), axis=1)]
                if self.method == "spearman":
                    c = stats.spearmanr(cdf.iloc[:, 0], cdf.iloc[:, 1])
                elif self.method == "kendall":
                    c = stats.kendalltau(cdf.iloc[:, 0], cdf.iloc[:, 1])
                else:
                    c = stats.pearsonr(cdf.iloc[:, 0], cdf.iloc[:, 1])

                cr[j, i] = c[0]
                cp[j, i] = c[1]
                cv[j, i] = cdf.iloc[:, [0, 1]].cov().iloc[0, 1]

        self.cr = cr
        self.cp = cp
        self.cv = cv

    def summary(self, cov=False, cutoff: float = 0, dec: int = 2) -> None:
        """
        Print correlations between numeric variables in a Pandas dataframe

        Parameters
        ----------
        cov : bool
            Show the covariance matrix if set to True
        cutoff : float
            Only show correlations larger than a threshold in absolute value
        dec : int
            Number of decimal places to use in rounding

        Examples
        --------
        import pandas as pd
        import pyrsm as rsm
        salary, salary_description = load_data(pkg="basics", name="salary")
        cr = rsm.correlation(salary[["salary", "yrs.since.phd", "yrs.service"]])
        cr.summary()
        """
        prn = "Correlation\n"
        prn += f"Data     : {self.name}\n"
        prn += f"Method   : {self.method}"
        print(prn)
        if len(self.vars) < 2:
            print("\n**Select two or more variables to calculate correlations**")
        else:
            ind = np.triu_indices(self.cr.shape[0])
            cn = list(self.data.columns)

            # correlations
            crs = pd.DataFrame(self.cr.round(dec).astype(str), columns=cn, index=cn)
            if cutoff > 0:
                crs.values[np.abs(self.cr) < cutoff] = ""
            crs.values[ind] = ""

            # p.values
            cps = pd.DataFrame(self.cp.round(dec).astype(str), columns=cn, index=cn)
            if cutoff > 0:
                cps.values[np.abs(self.cr) < cutoff] = ""
            cps.values[ind] = ""

            if len(cn) > 2:
                x = "x"
                y = "y"
            else:
                x = cn[0]
                y = cn[1]

            prn = f"Cutoff   : {cutoff}\n"
            prn += "Variables: " + ", ".join(cn) + "\n"
            prn += f"Null hyp.: variables {x} and {y} are not correlated\n"
            prn += f"Alt. hyp.: variables {x} and {y} are correlated\n"
            print(prn)
            print("Correlation matrix:")
            print(crs.iloc[1:, :-1])
            print("\np.values:")
            print(cps.iloc[1:, :-1])

            if cov:
                cvs = pd.DataFrame(self.cv.round(dec), columns=cn, index=cn).map(
                    lambda x: "{:,}".format(x)
                )
                if cutoff > 0:
                    cvs.values[np.abs(self.cr) < cutoff] = ""

                cvs.values[ind[0], ind[1]] = ""
                print("\nCovariance matrix:")
                print(cvs.iloc[1:, :-1])

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

        if figsize is None:
            figsize = (max(5, self.cr.shape[0]), max(self.cr.shape[0], 5))

        def cor_label(label, longest, ax_sub):
            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)
            # set font size to avoid exceeding boundaries
            fs = min(figsize[0], figsize[1])
            font = (80 * fs) / (len(longest) * self.cr.shape[0])
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

            fs = min(figsize[0], figsize[1])
            font = (50 * fs) / (len(str(rt)) * len(self.data.columns))
            font_star = (12 * fs) / len(self.data.columns)

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

        def cor_mat(data, cmat, pmat, dec=2, nobs=1000, figsize=None):
            cn = data.columns
            ncol = len(cn)
            longest = max(cn, key=len)
            s_size = 50 / len(self.data.columns)

            fs = min(figsize[0], figsize[1])
            s_size = (5 * fs) / len(self.data.columns)

            _, axes = plt.subplots(ncol, ncol, figsize=figsize)

            if nobs < data.shape[0] and nobs != np.inf and nobs != -1:
                data = data.copy().sample(nobs)

            for i in range(ncol):
                for j in range(ncol):
                    if i == j:
                        cor_label(cn[i], longest, axes[i, j])
                    elif i > j:
                        cor_plot(data[cn[i]], data[cn[j]], axes[i, j], s_size)
                    else:
                        cor_text(cmat[j, i], pmat[j, i], axes[i, j], dec=2)

            plt.subplots_adjust(wspace=0.04, hspace=0.04)

        cor_mat(self.data, self.cr, self.cp, dec=dec, nobs=nobs, figsize=figsize)
