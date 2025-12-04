from typing import Literal

import numpy as np
import polars as pl
from plotnine import aes, geom_histogram, ggplot, labs, theme_set
from scipy import stats

import pyrsm.basics.plotting_utils as pu
import pyrsm.basics.utils as bu
import pyrsm.basics.display_utils as du
from pyrsm.model.model import sig_stars
from pyrsm.utils import check_dataframe, ifelse


class single_mean:
    """
    A class to perform single-mean hypothesis testing

    Attributes
    ----------
    data : pl.DataFrame
        The input data for the hypothesis test as a Polars DataFrame. If a dictionary is provided, the key should be the name of the dataframe.
    var : str
        The variable/column name to test.
    alt_hyp : str
        The alternative hypothesis ('two-sided', 'greater', 'less').
    conf : float
        The confidence level for the test.
    comp_value : float
        The comparison value for the test.
    t_val : float
        The t-statistic value.
    p_val : float
        The p-value of the test.
    ci : tuple
        The confidence interval of the test.
    mean : float
        The mean of the variable.
    n : int
        The number of observations.
    n_missing : int
        The number of missing observations.
    sd : float
        The standard deviation of the variable.
    se : float
        The standard error of the variable.
    me : float
        The margin of error.
    diff : float
        The difference between the mean and the comparison value.
    df : int
        The degrees of freedom.

    Methods
    -------
    __init__(data, var, alt_hyp='two-sided', conf=0.95, comp_value=0)
        Initializes the single_mean class with the provided data and parameters.
    summary(dec=3)
        Prints a summary of the hypothesis test.
    plot()
        Plots the results of the hypothesis test.
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        var: str,
        alt_hyp: str = "two-sided",
        conf: float = 0.95,
        comp_value: float = 0,
    ):
        """
        Constructs all the necessary attributes for the single_mean object.

        Parameters
        ----------
        data : pl.DataFrame | dict[str, pl.DataFrame]
            The input data for the hypothesis test as a Polars DataFrame.
        var : str
            The variable/column name to test.
        alt_hyp : str, optional
            The alternative hypothesis ('two-sided', 'greater', 'less') (default is 'two-sided').
        conf : float, optional
            The confidence level for the test (default is 0.95).
        comp_value : float, optional
            The comparison value for the test (default is 0).
        """
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            incoming = data[self.name]
        else:
            incoming = data
            self.name = "Not provided"

        self.data = check_dataframe(incoming)
        self.var = var
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.comp_value = comp_value

        series = self.data[self.var]
        values = series.to_numpy()

        result = stats.ttest_1samp(
            a=values,
            popmean=self.comp_value,
            nan_policy="omit",
            alternative=self.alt_hyp,
        )

        self.t_val, self.p_val = result.statistic, result.pvalue
        self.ci = result.confidence_interval(confidence_level=conf)

        self.mean = np.nanmean(values)
        self.n = len(values)
        self.n_missing = int(series.null_count())

        self.sd = series.std(ddof=1)
        n_eff = self.n - self.n_missing
        self.se = self.sd / np.sqrt(n_eff) if n_eff > 0 else np.nan
        tscore = stats.t.ppf((1 + self.conf) / 2, n_eff - 1) if n_eff > 1 else np.nan

        self.me = (tscore * self.se).real
        self.diff = self.mean - self.comp_value
        self.df = n_eff - 1

    def summary(self, dec: int = 3) -> None:
        """
        Prints a summary of the hypothesis test.

        Parameters
        ----------
        dec : int, optional
            The number of decimal places to display (default is 3).
        """
        display = du.SummaryDisplay(
            header_func=self._summary_header,
            plain_func=lambda extra, d: self._summary_plain(d),
            styled_func=lambda extra, d: self._style_tables(d),
        )
        display.display(extra=False, dec=dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        print("Single mean test")
        print(f"Data      : {self.name}")
        print(f"Variables : {self.var}")
        print(f"Confidence: {self.conf}")
        print(f"Comparison: {self.comp_value}\n")
        print(f"Null hyp. : the mean of {self.var} is equal to {self.comp_value}")

        if self.alt_hyp == "less":
            alt_hyp = "less than"
        elif self.alt_hyp == "two-sided":
            alt_hyp = "not equal to"
        else:
            alt_hyp = "greater than"

        print(f"Alt. hyp. : the mean of {self.var} is {alt_hyp} {self.comp_value}\n")

    def _summary_plain(self, dec: int = 3) -> None:
        """Print plain text tables."""
        cl = bu.ci_label(self.alt_hyp, self.conf, dec=dec)

        table1 = pl.DataFrame({
            "mean": [round(self.mean, dec)],
            "n": [self.n],
            "n_missing": [self.n_missing],
            "sd": [round(self.sd, dec)],
            "se": [round(self.se, dec)],
            "me": [round(self.me, dec)],
        })

        p_val_str = ifelse(self.p_val < 0.001, "< .001", round(self.p_val, dec))
        table2 = pl.DataFrame({
            "diff": [round(self.diff, dec)],
            "se": [round(self.se, dec)],
            "t.value": [round(self.t_val, dec)],
            "p.value": [p_val_str],
            "df": [self.df],
            cl[0]: [round(self.ci[0], dec)],
            cl[1]: [round(self.ci[1], dec)],
            "": [sig_stars([self.p_val])[0]],
        })

        du.print_plain_tables(table1, table2)

    def _style_tables(self, dec: int = 3) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        cl = bu.ci_label(self.alt_hyp, self.conf, dec=dec)

        table1 = pl.DataFrame({
            "mean": [self.mean],
            "n": [self.n],
            "n_missing": [self.n_missing],
            "sd": [self.sd],
            "se": [self.se],
            "me": [self.me],
        })

        table2 = pl.DataFrame({
            "diff": [self.diff],
            "se": [self.se],
            "t.value": [self.t_val],
            "p.value": [du.format_pval(self.p_val, dec)],
            "df": [self.df],
            cl[0]: [self.ci[0]],
            cl[1]: [self.ci[1]],
            "": [sig_stars([self.p_val])[0]],
        })

        gt1 = du.style_table(
            table1,
            title="Descriptive Statistics",
            subtitle=f"Variable: {self.var}",
            number_cols=["mean", "sd", "se", "me"],
            integer_cols=["n", "n_missing"],
            dec=dec,
        )

        gt2 = du.style_table(
            table2,
            title="Hypothesis Test",
            subtitle=f"Comparison value: {self.comp_value}",
            number_cols=["diff", "se", "t.value", cl[0], cl[1]],
            integer_cols=["df"],
            dec=dec,
        )

        display(gt1)
        display(gt2)

    def plot(
        self,
        plots: Literal["hist", "sim"] = "hist",
        theme: Literal["modern", "publication", "minimal", "classic"] = "modern",
        backend: Literal["plotnine", "plotly"] = "plotnine",
    ):
        """
        Plots the results of the hypothesis test. If the 'hist' is selected a histogram
        of the numeric variable will be shown. The solid black line in the histogram shows
        the sample mean. The dashed black lines show the confidence interval around the
        sample mean. The solid red line shows the comparison value (i.e., the value under
        the null-hypothesis). If the red line does not fall within the confidence interval
        we can reject the null-hypothesis in favor of the alternative at the specified
        confidence level (e.g., 0.95).

        Parameters
        ----------
        plots : str
            The type of plot to generate (default is 'hist').
        theme : str
            The plotnine theme to use ('modern', 'publication', 'minimal', 'classic').
        backend : str
            Plotting backend to use ('plotnine' or 'plotly').

        Returns
        -------
        plotnine.ggplot or plotly.graph_objects.Figure
            The plot object (if backend is specified).
        """
        if backend == "plotly":
            return self._plot_plotly(plots)

        if plots == "hist":
            # Prepare data - convert polars to pandas for plotnine
            plot_data = self.data.select(self.var).drop_nulls().to_pandas()

            # Create plotnine histogram
            p = (
                ggplot(plot_data, aes(x=self.var))
                + geom_histogram(bins=10, fill=pu.PlotConfig.FILL)
                + labs(x="", y="Frequency")
                + pu.PlotConfig.theme()
            )

            # Add reference lines
            ref_lines = pu.ReferenceLine.ci_vlines(
                self.mean, self.ci[0], self.ci[1], self.comp_value
            )
            for line in ref_lines:
                p = p + line

            return p

        elif plots == "sim":
            print("Plot type not available yet")
            return None
        else:
            print("Invalid plot type")
            return None

    def _plot_plotly(self, plot_type: str):
        """
        Generate interactive plotly version of the plot.

        Parameters
        ----------
        plot_type : str
            The type of plot to generate.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive plotly figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not installed. Install with: uv add plotly")
            return None

        if plot_type == "hist":
            # Create histogram data - convert polars to numpy for plotly
            data_values = self.data.select(self.var).drop_nulls().to_series().to_numpy()

            fig = go.Figure()

            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=data_values,
                    nbinsx=30,
                    marker_color="slateblue",
                    opacity=0.7,
                    name=self.var,
                )
            )

            # Add reference lines
            fig.add_vline(
                x=self.comp_value,
                line_dash="solid",
                line_color="red",
                annotation_text=f"H₀: {self.comp_value}",
                annotation_position="top",
            )
            fig.add_vline(
                x=self.mean,
                line_dash="solid",
                line_color="black",
                annotation_text=f"Mean: {self.mean:.2f}",
                annotation_position="top",
            )
            fig.add_vline(
                x=self.ci[0], line_dash="dash", line_color="black", opacity=0.7
            )
            fig.add_vline(
                x=self.ci[1], line_dash="dash", line_color="black", opacity=0.7
            )

            # Update layout
            fig.update_layout(
                title=f"Distribution of {self.var}",
                xaxis_title=self.var,
                yaxis_title="Count",
                showlegend=False,
                hovermode="x unified",
            )

            return fig
        else:
            print(f"Plotly backend does not support plot type: {plot_type}")
            return None
