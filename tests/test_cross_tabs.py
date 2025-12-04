"""Tests for pyrsm.basics.cross_tabs module."""

import warnings
import io
import sys
import pytest
import polars as pl
import numpy as np
from pyrsm.basics import cross_tabs
import pyrsm as rsm


@pytest.fixture(scope="module")
def newspaper_data():
    """Load the newspaper dataset."""
    newspaper, _ = rsm.load_data(pkg="basics", name="newspaper")
    return newspaper


@pytest.fixture
def synthetic_crosstab_data():
    """Create synthetic data for cross-tabulation testing."""
    np.random.seed(42)
    n = 200
    # Create two categorical variables with known association
    var1 = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.3, 0.4, 0.3])
    # var2 is correlated with var1
    var2 = []
    for v in var1:
        if v == "Low":
            var2.append(np.random.choice(["A", "B"], p=[0.7, 0.3]))
        elif v == "Medium":
            var2.append(np.random.choice(["A", "B"], p=[0.5, 0.5]))
        else:
            var2.append(np.random.choice(["A", "B"], p=[0.3, 0.7]))
    return pl.DataFrame({"income": var1, "preference": var2})


@pytest.fixture
def independent_data():
    """Create data where variables are independent."""
    np.random.seed(42)
    n = 200
    var1 = np.random.choice(["X", "Y"], size=n, p=[0.5, 0.5])
    var2 = np.random.choice(["P", "Q"], size=n, p=[0.5, 0.5])
    return pl.DataFrame({"var1": var1, "var2": var2})


class TestCrossTabsBasic:
    """Basic tests for cross_tabs functionality."""

    def test_cross_tabs_basic(self, newspaper_data):
        """Test basic cross_tabs with newspaper data."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        assert ct.name == "newspaper"
        assert ct.var1 == "Income"
        assert ct.var2 == "Newspaper"
        assert "Total" in ct.observed.columns
        assert "Total" in ct.expected.columns

    def test_cross_tabs_observed_frequencies(self, synthetic_crosstab_data):
        """Test observed frequencies calculation."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        # Check total matches data size - use polars syntax
        total_row = ct.observed.filter(pl.col("income") == "Total")
        assert total_row["Total"].item() == 200

    def test_cross_tabs_expected_frequencies(self, synthetic_crosstab_data):
        """Test expected frequencies calculation."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        # Expected total should equal observed total - use polars syntax
        total_row = ct.expected.filter(pl.col("income") == "Total")
        assert total_row["Total"].item() == pytest.approx(200, abs=0.1)


class TestCrossTabsChiSquare:
    """Test chi-square test functionality."""

    def test_cross_tabs_chisq_test(self, newspaper_data):
        """Test chi-square test calculation."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        # chisq_test contains (chi2, p_value, dof, expected)
        assert len(ct.chisq_test) == 4
        assert ct.chisq_test[0] >= 0  # Chi-square is non-negative
        assert 0 <= ct.chisq_test[1] <= 1  # p-value is valid

    def test_cross_tabs_chisq_contributions(self, synthetic_crosstab_data):
        """Test chi-square contribution values."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        # All chi-square contributions should be non-negative
        # Use polars syntax - filter out Total row and drop Total column
        chisq_no_total = ct.chisq.filter(pl.col("income") != "Total").drop("Total")
        numeric_cols = [c for c in chisq_no_total.columns if c != "income"]
        for col in numeric_cols:
            assert all(v >= 0 for v in chisq_no_total[col].to_list())

    def test_cross_tabs_expected_low(self, synthetic_crosstab_data):
        """Test expected_low calculation."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        # expected_low should be [count_below_5, total_cells]
        assert len(ct.expected_low) == 2
        assert ct.expected_low[0] >= 0
        assert ct.expected_low[1] > 0


class TestCrossTabsPercentages:
    """Test percentage calculations."""

    def test_cross_tabs_perc_row(self, synthetic_crosstab_data):
        """Test row percentages."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        # Row totals should equal 1 (100%) - use polars syntax
        row_totals = ct.perc_row.filter(pl.col("income") != "Total")["Total"].to_list()
        assert all(t == pytest.approx(1.0, abs=0.001) for t in row_totals)

    def test_cross_tabs_perc_col(self, synthetic_crosstab_data):
        """Test column percentages."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        # Column totals should equal 1 (100%) - use polars syntax
        total_row = ct.perc_col.filter(pl.col("income") == "Total")
        cols_to_check = [c for c in total_row.columns if c not in ["income", "Total"]]
        for col in cols_to_check:
            assert total_row[col].item() == pytest.approx(1.0, abs=0.001)

    def test_cross_tabs_perc_total(self, synthetic_crosstab_data):
        """Test total percentages."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        # Total percentage should equal 1 (100%) - use polars syntax
        total_row = ct.perc.filter(pl.col("income") == "Total")
        assert total_row["Total"].item() == pytest.approx(1.0, abs=0.001)


class TestCrossTabsDevStd:
    """Test standardized deviation calculations."""

    def test_cross_tabs_dev_std(self, synthetic_crosstab_data):
        """Test standardized deviations exist."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        # dev_std should not have Total column
        assert "Total" not in ct.dev_std.columns
        # dev_std should not have Total in income column values
        income_values = ct.dev_std["income"].to_list()
        assert "Total" not in income_values


class TestCrossTabsInputFormats:
    """Test different input formats."""

    def test_cross_tabs_polars_input(self, synthetic_crosstab_data):
        """Test with polars DataFrame input."""
        ct = cross_tabs(
            data=synthetic_crosstab_data,
            var1="income",
            var2="preference",
        )
        assert "Total" in ct.observed.columns

    def test_cross_tabs_dict_input(self, synthetic_crosstab_data):
        """Test with dict input format."""
        ct = cross_tabs(
            data={"test_data": synthetic_crosstab_data},
            var1="income",
            var2="preference",
        )
        assert ct.name == "test_data"


class TestCrossTabsSummary:
    """Test summary output."""

    def test_summary_output_default(self, newspaper_data):
        """Test summary with default output tables."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        captured = io.StringIO()
        sys.stdout = captured
        ct.summary(dec=2)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Cross-tabs" in output
        assert "Income" in output
        assert "Newspaper" in output
        assert "Observed" in output
        assert "Expected" in output
        assert "Chi-squared" in output

    def test_summary_output_all_tables(self, newspaper_data):
        """Test summary with all output tables."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        captured = io.StringIO()
        sys.stdout = captured
        ct.summary(
            output=["observed", "expected", "chisq", "dev_std", "perc_row", "perc_col", "perc"],
            dec=2,
        )
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Observed" in output
        assert "Expected" in output
        assert "Contribution to chi-squared" in output
        assert "Deviation standardized" in output
        assert "Row percentages" in output
        assert "Column percentages" in output
        assert "Percentages" in output


class TestCrossTabsBaseline:
    """Baseline tests to capture outputs before plotnine conversion."""

    def test_baseline_tables(self, newspaper_data):
        """Capture baseline frequency/percentage tables."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )

        # Verify all tables exist and have correct structure
        assert isinstance(ct.observed, pl.DataFrame)
        assert isinstance(ct.expected, pl.DataFrame)
        assert isinstance(ct.chisq, pl.DataFrame)
        assert isinstance(ct.dev_std, pl.DataFrame)
        assert isinstance(ct.perc_row, pl.DataFrame)
        assert isinstance(ct.perc_col, pl.DataFrame)
        assert isinstance(ct.perc, pl.DataFrame)

        # Verify Total column exists where expected
        assert "Total" in ct.observed.columns
        assert "Total" in ct.expected.columns

    def test_baseline_chisq_stats(self, newspaper_data):
        """Capture baseline chi-square test statistics."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )

        # chisq_test contains (chi2, p_value, dof, expected)
        assert len(ct.chisq_test) == 4
        assert ct.chisq_test[0] >= 0  # Chi-square is non-negative
        assert 0 <= ct.chisq_test[1] <= 1  # p-value is valid
        assert ct.chisq_test[2] > 0  # degrees of freedom > 0

    def test_baseline_plot_observed(self, newspaper_data, baseline_plot_dir):
        """Save baseline observed plot (now using plotnine)."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["observed"])

        out_file = baseline_plot_dir / "cross_tabs_observed_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_expected(self, newspaper_data, baseline_plot_dir):
        """Save baseline expected plot (now using plotnine)."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["expected"])

        out_file = baseline_plot_dir / "cross_tabs_expected_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_chisq(self, newspaper_data, baseline_plot_dir):
        """Save baseline chi-square contribution plot (now using plotnine)."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["chisq"])

        out_file = baseline_plot_dir / "cross_tabs_chisq_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_dev_std(self, newspaper_data, baseline_plot_dir):
        """Save baseline standardized deviation plot (now using plotnine)."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["dev_std"])

        out_file = baseline_plot_dir / "cross_tabs_dev_std_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_perc_col(self, newspaper_data, baseline_plot_dir):
        """Save baseline column percentage plot (now using plotnine)."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["perc_col"])

        out_file = baseline_plot_dir / "cross_tabs_perc_col_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_perc_row(self, newspaper_data, baseline_plot_dir):
        """Save baseline row percentage plot (now using plotnine)."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["perc_row"])

        out_file = baseline_plot_dir / "cross_tabs_perc_row_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000


class TestCrossTabsPlot:
    """Test plot functionality."""

    def test_plot_observed(self, newspaper_data):
        """Test observed plot generation."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["observed"])
        assert p is not None

    def test_plot_expected(self, newspaper_data):
        """Test expected plot generation."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["expected"])
        assert p is not None

    def test_plot_chisq(self, newspaper_data):
        """Test chi-square contribution plot generation."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["chisq"])
        assert p is not None

    def test_plot_dev_std(self, newspaper_data):
        """Test standardized deviation plot generation."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["dev_std"])
        assert p is not None

    def test_plot_perc_col(self, newspaper_data):
        """Test column percentage plot (default)."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots="perc_col")
        assert p is not None

    def test_plot_perc_row(self, newspaper_data):
        """Test row percentage plot."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["perc_row"])
        assert p is not None

    def test_plot_perc(self, newspaper_data):
        """Test total percentage plot."""
        ct = cross_tabs(
            data={"newspaper": newspaper_data},
            var1="Income",
            var2="Newspaper",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = ct.plot(plots=["perc"])
        assert p is not None
