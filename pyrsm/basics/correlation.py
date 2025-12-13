from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from pyrsm.model.model import sig_stars
from pyrsm.utils import check_dataframe
import pyrsm.basics.display_utils as du


class correlation:
    """
    Calculate correlations between numeric variables in a Polars dataframe

    Parameters
    ----------
    data : Polars dataframe with numeric variables

    Returns
    -------
    Correlation object with two key attributes
    cr: Correlation matrix
    cp: p.value matrix
    cv: Covariance matrix

    Examples
    --------
    import pyrsm as rsm
    salary, salary_description = rsm.load_data(pkg="basics", name="salary")
    cr = rsm.correlation(salary.select(["salary", "yrs.since.phd", "yrs.service"]))
    cr.cr
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        vars: list[str] | None = [],
        method: str = "pearson",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)
        self.vars = vars
        if len(self.vars) == 0:
            self.vars = [col for col, dtype in self.data.schema.items() if dtype.is_numeric()]

        self.data = self.data.select(self.vars)
        self.method = method

        ncol = self.data.shape[1]
        cr = np.zeros([ncol, ncol])
        cp = cr.copy()
        cv = cr.copy()
        for i in range(ncol - 1):
            for j in range(i + 1, ncol):
                x = self.data.select(self.vars[i]).to_series().to_numpy()
                y = self.data.select(self.vars[j]).to_series().to_numpy()
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                if x.dtype == bool:
                    x = x.astype(int)
                if y.dtype == bool:
                    y = y.astype(int)
                if self.method == "spearman":
                    c = stats.spearmanr(x, y)
                elif self.method == "kendall":
                    c = stats.kendalltau(x, y)
                else:
                    c = stats.pearsonr(x, y)

                cr[j, i] = c[0]
                cp[j, i] = c[1]
                cv[j, i] = np.cov(x, y, ddof=1)[0, 1]
                cr[i, j] = cr[j, i]
                cp[i, j] = cp[j, i]
                cv[i, j] = cv[j, i]

        self.cr = cr
        self.cp = cp
        self.cv = cv

    def summary(self, cov=False, cutoff: float = 0, dec: int = 2, plain: bool = True) -> None:
        """
        Print correlations between numeric variables in a Polars dataframe

        Parameters
        ----------
        cov : bool
            Show the covariance matrix if set to True
        cutoff : float
            Only show correlations larger than a threshold in absolute value
        dec : int
            Number of decimal places to use in rounding
        plain : bool
            If True (default), print plain text output. If False and running
            in a Jupyter notebook, use styled table output.

        Examples
        --------
        import pyrsm as rsm
        salary, salary_description = rsm.load_data(pkg="basics", name="salary")
        cr = rsm.correlation(salary.select(["salary", "yrs.since.phd", "yrs.service"]))
        cr.summary()
        """
        self._summary_header()
        if len(self.vars) < 2:
            print("\n**Select two or more variables to calculate correlations**")
            return

        self._print_hypothesis_info(cutoff)

        if not plain and du.is_notebook():
            self._style_tables(cov=cov, cutoff=cutoff, dec=dec)
        else:
            self._summary_plain(cov=cov, cutoff=cutoff, dec=dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        prn = "Correlation\n"
        prn += f"Data     : {self.name}\n"
        prn += f"Method   : {self.method}"
        print(prn)

    def _print_hypothesis_info(self, cutoff: float) -> None:
        """Print hypothesis and variable information."""
        cn = list(self.data.columns)
        if len(cn) > 2:
            x, y = "x", "y"
        else:
            x, y = cn[0], cn[1]

        prn = f"Cutoff   : {cutoff}\n"
        prn += "Variables: " + ", ".join(cn) + "\n"
        prn += f"Null hyp.: variables {x} and {y} are not correlated\n"
        prn += f"Alt. hyp.: variables {x} and {y} are correlated\n"
        print(prn)

    def _build_matrix_displays(self, cutoff: float, dec: int):
        """Build correlation, p-value, and covariance display DataFrames."""
        ind = np.triu_indices(self.cr.shape[0])
        cn = list(self.data.columns)

        # Build correlation matrix
        crs_arr = self.cr.round(dec).astype(str)
        if cutoff > 0:
            crs_arr[np.abs(self.cr) < cutoff] = ""
        crs_arr[ind] = ""

        # Build p-value matrix
        cps_arr = self.cp.round(dec).astype(str)
        if cutoff > 0:
            cps_arr[np.abs(self.cr) < cutoff] = ""
        cps_arr[ind] = ""

        # Create polars DataFrames
        crs_display = pl.DataFrame(
            {cn[j]: crs_arr[1:, j] for j in range(len(cn) - 1)}
        ).with_columns(pl.Series("", cn[1:]).alias(""))
        crs_display = crs_display.select([""] + cn[:-1])

        cps_display = pl.DataFrame(
            {cn[j]: cps_arr[1:, j] for j in range(len(cn) - 1)}
        ).with_columns(pl.Series("", cn[1:]).alias(""))
        cps_display = cps_display.select([""] + cn[:-1])

        # Build covariance matrix
        cvs_arr = np.round(self.cv, dec)
        cvs_str = np.array([["{:,}".format(v) for v in row] for row in cvs_arr])
        if cutoff > 0:
            cvs_str[np.abs(self.cr) < cutoff] = ""
        cvs_str[ind[0], ind[1]] = ""

        cvs_display = pl.DataFrame(
            {cn[j]: cvs_str[1:, j] for j in range(len(cn) - 1)}
        ).with_columns(pl.Series("", cn[1:]).alias(""))
        cvs_display = cvs_display.select([""] + cn[:-1])

        return crs_display, cps_display, cvs_display

    def _summary_plain(self, cov: bool = False, cutoff: float = 0, dec: int = 2) -> None:
        """Print plain text tables."""
        crs_display, cps_display, cvs_display = self._build_matrix_displays(cutoff, dec)

        with pl.Config(
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            fmt_str_lengths=100,
        ):
            print("Correlation matrix:")
            print(crs_display)
            print("\np.values:")
            print(cps_display)

            if cov:
                print("\nCovariance matrix:")
                print(cvs_display)

    def _style_tables(self, cov: bool = False, cutoff: float = 0, dec: int = 2) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        crs_display, cps_display, cvs_display = self._build_matrix_displays(cutoff, dec)

        gt1 = du.style_table(
            crs_display,
            title="Correlation Matrix",
            subtitle=f"Method: {self.method}",
        )
        gt2 = du.style_table(
            cps_display,
            title="P-values",
            subtitle="",
        )

        display(gt1)
        display(gt2)

        if cov:
            gt3 = du.style_table(
                cvs_display,
                title="Covariance Matrix",
                subtitle="",
            )
            display(gt3)

    def plot(self, nobs: int = 1000, dec: int = 2, figsize: tuple[float, float] = None):
        """
        Plot scatter matrix of correlations between numeric variables.

        Displays a matrix with:
        - Diagonal: variable names
        - Lower triangle: scatter plots with regression lines
        - Upper triangle: correlation coefficients with significance stars

        Parameters
        ----------
        nobs : int
            Number of observations to use for the scatter plots. The default
            value is 1,000. To use all observations in the plots, use nobs=-1
        dec : int
            Number of decimal places to use in rounding
        figsize : tuple
            A tuple that determines the figure size. If None, size is
            determined based on the number of numeric variables in the data

        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects

        Examples
        --------
        import pyrsm as rsm
        salary, _ = rsm.load_data(pkg="basics", name="salary")
        cr = rsm.correlation(salary.select(["salary", "yrs_since_phd", "yrs_service"]))
        cr.plot(figsize=(7, 7))
        """
        if figsize is None:
            figsize = (max(5, self.cr.shape[0]), max(self.cr.shape[0], 5))

        def cor_label(label, longest, ax_sub):
            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)
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
            # Remove NaN values for regression
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_clean = x_data[mask]
            y_clean = y_data[mask]

            # Scatter plot
            ax_sub.scatter(x_clean, y_clean, alpha=0.3, color="black", s=s_size)

            # Regression line
            if len(x_clean) > 1:
                coeffs = np.polyfit(x_clean, y_clean, 1)
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_line = np.polyval(coeffs, x_line)
                ax_sub.plot(x_line, y_line, color="blue")

            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)

        cn = list(self.data.columns)
        ncol = len(cn)
        longest = max(cn, key=len)

        fs = min(figsize[0], figsize[1])
        s_size = (5 * fs) / len(self.data.columns)

        fig, axes = plt.subplots(ncol, ncol, figsize=figsize)

        # Sample data if needed
        nrows = self.data.shape[0]
        if nobs < nrows and nobs != np.inf and nobs != -1:
            indices = np.random.choice(nrows, size=nobs, replace=False)
            data_np = {col: self.data[col].to_numpy()[indices] for col in cn}
        else:
            data_np = {col: self.data[col].to_numpy() for col in cn}

        for i in range(ncol):
            for j in range(ncol):
                if i == j:
                    cor_label(cn[i], longest, axes[i, j])
                elif i > j:
                    cor_plot(
                        data_np[cn[i]],
                        data_np[cn[j]],
                        axes[i, j],
                        s_size,
                    )
                else:
                    cor_text(self.cr[j, i], self.cp[j, i], axes[i, j], dec=dec)

        plt.subplots_adjust(wspace=0.04, hspace=0.04)
        return fig, axes
