from typing import Union

import numpy as np
import polars as pl
from scipy import stats
from plotnine import (
    aes,
    geom_col,
    ggplot,
    ggtitle,
    labs,
    position_dodge,
    position_stack,
    theme_bw,
)

from pyrsm.utils import check_dataframe, ifelse
import pyrsm.basics.display_utils as du
import pyrsm.basics.plotting_utils as pu


class cross_tabs:
    """
    Calculate a Chi-square test between two categorical variables contained
    in a Polars dataframe

    Parameters
    ----------
    data : Polars dataframe with categorical variables or a
        dictionary with a single dataframe as the value and the
        name of the dataframe as the key
    var1: String; Name of the first categorical variable
    var2: String; Name of the second categorical variable

    Returns
    -------
    Cross object with several attributes
    data: Original dataframe
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
    newspaper, newspapar_description = rsm.load_data(pkg="basics", name="newspaper")
    ct = rsm.cross_tabs(newspaper, "Income", "Newspaper")
    ct.expected
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        var1: str,
        var2: str,
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)
        self.var1 = var1
        self.var2 = var2

        # Get unique levels for each variable
        var1_col = self.data[var1]
        var2_col = self.data[var2]

        if var1_col.dtype == pl.Categorical:
            var1_levels = var1_col.cat.get_categories().to_list()
        else:
            var1_levels = sorted(self.data[var1].unique().drop_nulls().to_list())

        if var2_col.dtype == pl.Categorical:
            var2_levels = var2_col.cat.get_categories().to_list()
        else:
            var2_levels = sorted(self.data[var2].unique().drop_nulls().to_list())

        # Build crosstab using polars pivot
        counts = (
            self.data.select([var1, var2])
            .drop_nulls()
            .group_by([var1, var2])
            .len()
        )

        # Pivot to get crosstab format
        pivot_df = counts.pivot(on=var2, index=var1, values="len").fill_null(0)

        # Ensure all var2 levels are present and in order
        for lev in var2_levels:
            if str(lev) not in pivot_df.columns:
                pivot_df = pivot_df.with_columns(pl.lit(0).alias(str(lev)))

        # Sort by var1 and select columns in order
        pivot_df = pivot_df.sort(var1)
        obs_cols = [var1] + [str(lev) for lev in var2_levels]
        pivot_df = pivot_df.select([c for c in obs_cols if c in pivot_df.columns])

        # Cast var1 to string for consistency
        pivot_df = pivot_df.with_columns(pl.col(var1).cast(pl.Utf8))

        # Add row totals - use string column names
        var2_str_levels = [str(lev) for lev in var2_levels]
        pivot_df = pivot_df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in var2_str_levels]).alias("Total")
        )

        # Cast all numeric columns to Int64 for consistency
        for col in var2_str_levels + ["Total"]:
            pivot_df = pivot_df.with_columns(pl.col(col).cast(pl.Int64))

        # Add column totals row
        col_totals = {var1: "Total"}
        for col in var2_str_levels:
            col_totals[col] = int(pivot_df[col].sum())
        col_totals["Total"] = int(pivot_df["Total"].sum())
        totals_row = pl.DataFrame([col_totals])
        self.observed = pl.concat([pivot_df, totals_row])

        # Store string versions for later use
        var1_str_levels = [str(lev) for lev in var1_levels]

        # Extract numeric part for chi-square test (without margins)
        obs_matrix = np.array([
            [pivot_df.filter(pl.col(var1) == str(r))[str(c)].item() for c in var2_levels]
            for r in var1_levels
        ])

        self.chisq_test = stats.chi2_contingency(obs_matrix, correction=False)
        expected_matrix = self.chisq_test[3]

        self.expected_low = [
            int((expected_matrix < 5).sum()),
            expected_matrix.shape[0] * expected_matrix.shape[1],
        ]

        # Build expected DataFrame
        exp_data = {var1: var1_str_levels}
        for j, col in enumerate(var2_str_levels):
            exp_data[col] = expected_matrix[:, j].tolist()
        expected_df = pl.DataFrame(exp_data)
        expected_df = expected_df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in var2_str_levels]).alias("Total")
        )
        exp_totals = {var1: "Total"}
        for col in var2_str_levels:
            exp_totals[col] = expected_df[col].sum()
        exp_totals["Total"] = expected_df["Total"].sum()
        exp_totals_row = pl.DataFrame([exp_totals])
        self.expected = pl.concat([expected_df, exp_totals_row])

        # Build chisq DataFrame
        chisq_matrix = (obs_matrix - expected_matrix) ** 2 / expected_matrix
        chisq_data = {var1: var1_str_levels}
        for j, col in enumerate(var2_str_levels):
            chisq_data[col] = chisq_matrix[:, j].tolist()
        chisq_df = pl.DataFrame(chisq_data)
        chisq_df = chisq_df.with_columns(
            pl.sum_horizontal([pl.col(c) for c in var2_str_levels]).alias("Total")
        )
        chisq_totals = {var1: "Total"}
        for col in var2_str_levels:
            chisq_totals[col] = chisq_df[col].sum()
        chisq_totals["Total"] = chisq_df["Total"].sum()
        chisq_totals_row = pl.DataFrame([chisq_totals])
        self.chisq = pl.concat([chisq_df, chisq_totals_row])

        # Build dev_std DataFrame (without totals)
        dev_std_matrix = (obs_matrix - expected_matrix) / np.sqrt(expected_matrix)
        dev_std_data = {var1: var1_str_levels}
        for j, col in enumerate(var2_str_levels):
            dev_std_data[col] = dev_std_matrix[:, j].tolist()
        self.dev_std = pl.DataFrame(dev_std_data)

        # Build percentage DataFrames
        # perc_row: each row sums to 1
        perc_row_data = {var1: var1_str_levels + ["Total"]}
        for col in var2_str_levels + ["Total"]:
            perc_row_data[col] = [
                self.observed.filter(pl.col(var1) == r)[col].item() /
                self.observed.filter(pl.col(var1) == r)["Total"].item()
                for r in var1_str_levels + ["Total"]
            ]
        self.perc_row = pl.DataFrame(perc_row_data)

        # perc_col: each column sums to 1
        perc_col_data = {var1: var1_str_levels + ["Total"]}
        col_total_row = self.observed.filter(pl.col(var1) == "Total")
        for col in var2_str_levels + ["Total"]:
            col_sum = col_total_row[col].item()
            perc_col_data[col] = [
                self.observed.filter(pl.col(var1) == r)[col].item() / col_sum
                for r in var1_str_levels + ["Total"]
            ]
        self.perc_col = pl.DataFrame(perc_col_data)

        # perc: all cells sum to 1
        grand_total = col_total_row["Total"].item()
        perc_data = {var1: var1_str_levels + ["Total"]}
        for col in var2_str_levels + ["Total"]:
            perc_data[col] = [
                self.observed.filter(pl.col(var1) == r)[col].item() / grand_total
                for r in var1_str_levels + ["Total"]
            ]
        self.perc = pl.DataFrame(perc_data)

        # Store for plotting (use string versions)
        self._var1_levels = var1_str_levels
        self._var2_levels = var2_str_levels

    def summary(self, output: list[str] = ["observed", "expected"], dec: int = 2) -> None:
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
        newspaper, newspapar_description = rsm.load_data(pkg="basics", name="newspaper")
        ct = rsm.cross_tabs(newspaper)
        ct.summary()
        """
        output = ifelse(isinstance(output, str), [output], output)
        self._summary_header()

        if du.is_notebook():
            self._style_tables(output=output, dec=dec)
        else:
            self._summary_plain(output=output, dec=dec)

        self._summary_footer(dec=dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        print(f"""
Cross-tabs
Data     : {self.name}
Variables: {self.var1}, {self.var2}
Null hyp : There is no association between {self.var1} and {self.var2}
Alt. hyp : There is an association between {self.var1} and {self.var2}""")

    def _summary_footer(self, dec: int = 2) -> None:
        """Print the chi-squared test results."""
        p_val_str = ifelse(self.chisq_test[1] < 0.001, "< .001", round(self.chisq_test[1], dec))
        print(f"""
Chi-squared: {round(self.chisq_test[0], dec)} df({int(self.chisq_test[2])}), p.value {p_val_str}
{100 * round(self.expected_low[0] / self.expected_low[1], dec)}% of cells have expected values below 5
""")

    def _summary_plain(self, output: list[str], dec: int = 2) -> None:
        """Print plain text tables."""
        with pl.Config(
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            fmt_str_lengths=100,
        ):
            if "observed" in output:
                print("\nObserved:\n")
                print(self.observed)

            if "expected" in output:
                print("\nExpected: (row total x column total) / total\n")
                exp_rounded = self.expected.select([
                    pl.col(self.var1),
                    *[pl.col(c).cast(pl.Float64).round(dec) for c in self._var2_levels + ["Total"]]
                ])
                print(exp_rounded)

            if "chisq" in output:
                print("\nContribution to chi-squared: (o - e)^2 / e\n")
                chisq_rounded = self.chisq.select([
                    pl.col(self.var1),
                    *[pl.col(c).cast(pl.Float64).round(dec) for c in self._var2_levels + ["Total"]]
                ])
                print(chisq_rounded)

            if "dev_std" in output:
                print("\nDeviation standardized: (o - e) / sqrt(e)\n")
                dev_rounded = self.dev_std.select([
                    pl.col(self.var1),
                    *[pl.col(c).round(dec) for c in self._var2_levels]
                ])
                print(dev_rounded)

            if "perc_row" in output:
                print("\nRow percentages:\n")
                perc_row_pct = self.perc_row.select([
                    pl.col(self.var1),
                    *[(pl.col(c) * 100).round(dec).cast(pl.Utf8) + "%" for c in self._var2_levels + ["Total"]]
                ])
                print(perc_row_pct)

            if "perc_col" in output:
                print("\nColumn percentages:\n")
                perc_col_pct = self.perc_col.select([
                    pl.col(self.var1),
                    *[(pl.col(c) * 100).round(dec).cast(pl.Utf8) + "%" for c in self._var2_levels + ["Total"]]
                ])
                print(perc_col_pct)

            if "perc" in output:
                print("\nPercentages:\n")
                perc_pct = self.perc.select([
                    pl.col(self.var1),
                    *[(pl.col(c) * 100).round(dec).cast(pl.Utf8) + "%" for c in self._var2_levels + ["Total"]]
                ])
                print(perc_pct)

    def _style_tables(self, output: list[str], dec: int = 2) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        if "observed" in output:
            gt = du.style_table(
                self.observed, title="Observed Frequencies", subtitle=""
            )
            display(gt)

        if "expected" in output:
            exp_rounded = self.expected.select([
                pl.col(self.var1),
                *[pl.col(c).cast(pl.Float64).round(dec) for c in self._var2_levels + ["Total"]]
            ])
            gt = du.style_table(
                exp_rounded,
                title="Expected Frequencies",
                subtitle="(row total x column total) / total",
            )
            display(gt)

        if "chisq" in output:
            chisq_rounded = self.chisq.select([
                pl.col(self.var1),
                *[pl.col(c).cast(pl.Float64).round(dec) for c in self._var2_levels + ["Total"]]
            ])
            gt = du.style_table(
                chisq_rounded,
                title="Chi-squared Contributions",
                subtitle="(o - e)² / e",
            )
            display(gt)

        if "dev_std" in output:
            dev_rounded = self.dev_std.select([
                pl.col(self.var1),
                *[pl.col(c).round(dec) for c in self._var2_levels]
            ])
            gt = du.style_table(
                dev_rounded,
                title="Standardized Deviations",
                subtitle="(o - e) / √e",
            )
            display(gt)

        if "perc_row" in output:
            perc_row_pct = self.perc_row.select([
                pl.col(self.var1),
                *[(pl.col(c) * 100).round(dec).cast(pl.Utf8) + "%" for c in self._var2_levels + ["Total"]]
            ])
            gt = du.style_table(perc_row_pct, title="Row Percentages", subtitle="")
            display(gt)

        if "perc_col" in output:
            perc_col_pct = self.perc_col.select([
                pl.col(self.var1),
                *[(pl.col(c) * 100).round(dec).cast(pl.Utf8) + "%" for c in self._var2_levels + ["Total"]]
            ])
            gt = du.style_table(perc_col_pct, title="Column Percentages", subtitle="")
            display(gt)

        if "perc" in output:
            perc_pct = self.perc.select([
                pl.col(self.var1),
                *[(pl.col(c) * 100).round(dec).cast(pl.Utf8) + "%" for c in self._var2_levels + ["Total"]]
            ])
            gt = du.style_table(perc_pct, title="Percentages", subtitle="")
            display(gt)

    def plot(self, plots: list[str] = "perc_col", **kwargs):
        """
        Plot of cross-tabulation results

        Parameters
        ----------
        plots : list of tables to show
            Options include "observed" (observed frequencies),
            "expected" (expected frequencies), "chisq" (chi-square values)
            for each cell, "dev_std" (standardized deviations from expected)
            "perc_row" (percentages conditioned by row), "perc_col"
            (percentages conditioned by column), "perc" (percentages by the
            total number of observations). The default value is "perc_col"
        **kwargs : Named arguments to be passed to plotting functions

        Returns
        -------
        plotnine.ggplot
            A plotnine ggplot object

        Examples
        --------
        import pyrsm as rsm
        newspaper, newspapar_description = rsm.load_data(pkg="basics", name="newspaper")
        ct = rsm.cross_tabs(newspaper, "Income", "Newspaper")
        ct.plot()
        """
        plots = ifelse(isinstance(plots, str), [plots], plots)

        def _reshape_to_long(df, value_name):
            """Convert wide polars DataFrame to long format for plotting."""
            rows = df.filter(pl.col(self.var1) != "Total")
            data_list = []
            for r in self._var1_levels:
                row_data = rows.filter(pl.col(self.var1) == r)
                for c in self._var2_levels:
                    data_list.append({
                        self.var1: r,
                        self.var2: c,
                        value_name: row_data[c].item(),
                    })
            return pl.DataFrame(data_list)

        def _stacked_bar(df, title, value_name):
            """Create stacked bar chart (normalized to percentages)."""
            long_df = _reshape_to_long(df, value_name)
            # Normalize within each var2 group
            totals = long_df.group_by(self.var2).agg(
                pl.col(value_name).sum().alias("total")
            )
            long_df = long_df.join(totals, on=self.var2)
            long_df = long_df.with_columns(
                (pl.col(value_name) / pl.col("total") * 100).alias("pct")
            )
            p = (
                ggplot(long_df, aes(x=self.var2, y="pct", fill=self.var1))
                + geom_col(position=position_stack(), alpha=0.8)
                + labs(x=self.var2, y="Percentage", fill=self.var1)
                + ggtitle(title)
                + theme_bw()
            )
            return p

        def _grouped_bar(df, title, value_name):
            """Create grouped (dodged) bar chart."""
            long_df = _reshape_to_long(df, value_name)
            p = (
                ggplot(long_df, aes(x=self.var2, y=value_name, fill=self.var1))
                + geom_col(position=position_dodge(), alpha=0.8)
                + labs(x=self.var2, y=value_name, fill=self.var1)
                + ggtitle(title)
                + theme_bw()
            )
            return p

        p = None

        if "observed" in plots:
            p = _stacked_bar(self.observed, "Observed frequencies", "count")

        if "expected" in plots:
            p = _stacked_bar(self.expected, "Expected frequencies", "expected")

        if "chisq" in plots:
            p = _grouped_bar(self.chisq, "Contribution to chi-squared statistic", "chisq")

        if "dev_std" in plots:
            long_df = _reshape_to_long(self.dev_std, "stdev")
            p = (
                ggplot(long_df, aes(x=self.var2, y="stdev", fill=self.var1))
                + geom_col(position=position_dodge(), alpha=0.8)
                + labs(x=self.var2, y="Std. Deviation", fill=self.var1)
                + ggtitle("Deviation standardized")
                + theme_bw()
            )
            for line in pu.ReferenceLine.significance_levels():
                p = p + line

        if "perc_col" in plots:
            long_df = _reshape_to_long(self.perc_col, "pct")
            long_df = long_df.with_columns((pl.col("pct") * 100).alias("pct"))
            p = (
                ggplot(long_df, aes(x=self.var2, y="pct", fill=self.var1))
                + geom_col(position=position_dodge(), alpha=0.8)
                + labs(x=self.var2, y="Column %", fill=self.var1)
                + ggtitle("Column percentages")
                + theme_bw()
            )

        if "perc_row" in plots:
            long_df = _reshape_to_long(self.perc_row, "pct")
            long_df = long_df.with_columns((pl.col("pct") * 100).alias("pct"))
            p = (
                ggplot(long_df, aes(x=self.var2, y="pct", fill=self.var1))
                + geom_col(position=position_dodge(), alpha=0.8)
                + labs(x=self.var2, y="Row %", fill=self.var1)
                + ggtitle("Row percentages")
                + theme_bw()
            )

        if "perc" in plots:
            long_df = _reshape_to_long(self.perc, "pct")
            long_df = long_df.with_columns((pl.col("pct") * 100).alias("pct"))
            p = (
                ggplot(long_df, aes(x=self.var2, y="pct", fill=self.var1))
                + geom_col(position=position_dodge(), alpha=0.8)
                + labs(x=self.var2, y="Table %", fill=self.var1)
                + ggtitle("Table percentages")
                + theme_bw()
            )

        return p
