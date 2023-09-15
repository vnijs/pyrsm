import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from pyrsm.model.model import sig_stars
from typing import Optional, Union


class correlation:
    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        figsize: Optional[tuple[float, float]] = None,
    ) -> None:
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

        Examples
        --------
        import pandas as pd
        import pyrsm as rsm
        rsm.load_data(pkg="basics", name="salary", dct=globals())
        cr = rsm.correlation(salary[["salary", "yrs.since.phd", "yrs.service"]])
        c.cr
        """
        data = data.copy()
        isNum = [
            col
            for col in data.columns
            if pd.api.types.is_numeric_dtype(data[col].dtype)
        ]
        data = data[isNum]

        ncol = data.shape[1]
        cr = np.zeros([ncol, ncol])
        cp = cr.copy()
        for i in range(ncol - 1):
            for j in range(i + 1, ncol):
                cdf = data.iloc[:, [i, j]]
                # pairwise deletion
                cdf = cdf[~np.any(np.isnan(cdf), axis=1)]
                c = stats.pearsonr(cdf.iloc[:, 0], cdf.iloc[:, 1])
                cr[j, i] = c[0]
                cp[j, i] = c[1]

        self.data = data
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
        cn = self.data.columns[:-1]
        indn = self.data.columns[1:]

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

        cn = self.data.columns
        if len(cn) > 2:
            x = "x"
            y = "y"
        else:
            x = cn[0]
            y = cn[1]

        prn = "Correlation\n"
        prn += "Variables: " + ", ".join(list(self.data.columns)) + "\n"
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
            font = (80 * self.fig_size) / (len(longest) * len(self.data.columns))
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

            font = (50 * self.fig_size) / (len(str(rt)) * len(self.data.columns))
            font_star = (12 * self.fig_size) / len(self.data.columns)

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

            if figsize is None:
                figsize = (max(5, cmat.shape[0]), max(cmat.shape[0], 5))

            self.fig_size = min(figsize[0], figsize[1])
            s_size = (5 * self.fig_size) / len(self.data.columns)

            _, axes = plt.subplots(ncol, ncol, figsize=figsize)

            if nobs < data.shape[0] and nobs != np.Inf and nobs != -1:
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
