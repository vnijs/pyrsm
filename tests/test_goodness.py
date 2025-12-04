"""Tests for pyrsm.basics.goodness module."""

import warnings
import io
import sys
import pytest
import polars as pl
import numpy as np
from pyrsm.basics import goodness
import pyrsm as rsm


@pytest.fixture(scope="module")
def newspaper_data():
    """Load the newspaper dataset."""
    newspaper, _ = rsm.load_data(pkg="basics", name="newspaper")
    return newspaper


@pytest.fixture
def synthetic_categorical_data():
    """Create synthetic categorical data for testing."""
    np.random.seed(42)
    # Create data with known frequencies
    # A: 40, B: 30, C: 20, D: 10
    values = ["A"] * 40 + ["B"] * 30 + ["C"] * 20 + ["D"] * 10
    np.random.shuffle(values)
    return pl.DataFrame({"category": values})


@pytest.fixture
def uniform_data():
    """Create data with uniform distribution."""
    values = ["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25
    return pl.DataFrame({"category": values})


class TestGoodnessBasic:
    """Basic tests for goodness functionality."""

    def test_goodness_uniform_probs(self, newspaper_data):
        """Test goodness of fit with uniform probabilities."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        assert gf.name == "newspaper"
        assert gf.var == "Income"
        # With no probs specified, should be uniform
        assert all(p == pytest.approx(1 / gf.nlev) for p in gf.probs)
        assert len(gf.observed.columns) > 0
        assert len(gf.expected.columns) > 0

    def test_goodness_custom_probs(self, synthetic_categorical_data):
        """Test goodness with custom probabilities."""
        gf = goodness(
            data=synthetic_categorical_data,
            var="category",
            probs=(0.4, 0.3, 0.2, 0.1),
        )
        assert gf.probs == (0.4, 0.3, 0.2, 0.1)

    def test_goodness_observed_frequencies(self, synthetic_categorical_data):
        """Test that observed frequencies are correct."""
        gf = goodness(
            data=synthetic_categorical_data,
            var="category",
        )
        # Check total - use polars item() to get scalar
        assert gf.observed["Total"].item() == 100


class TestGoodnessInputFormats:
    """Test different input formats."""

    def test_goodness_polars_input(self, synthetic_categorical_data):
        """Test with polars DataFrame input."""
        gf = goodness(
            data=synthetic_categorical_data,
            var="category",
        )
        assert gf.nlev == 4

    def test_goodness_dict_input(self, synthetic_categorical_data):
        """Test with dict input format."""
        gf = goodness(
            data={"test_data": synthetic_categorical_data},
            var="category",
        )
        assert gf.name == "test_data"


class TestGoodnessChiSquare:
    """Test chi-square calculations."""

    def test_goodness_chisq_values(self, synthetic_categorical_data):
        """Test chi-square contribution values exist."""
        gf = goodness(
            data=synthetic_categorical_data,
            var="category",
        )
        assert "Total" in gf.chisq.columns
        assert gf.chisq["Total"].item() >= 0  # Chi-square is non-negative

    def test_goodness_stdev_values(self, synthetic_categorical_data):
        """Test standardized deviation values exist."""
        gf = goodness(
            data=synthetic_categorical_data,
            var="category",
        )
        assert len(gf.stdev.columns) == gf.nlev


class TestGoodnessSummary:
    """Test summary output."""

    def test_summary_output_default(self, newspaper_data):
        """Test that summary produces output with default settings."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        captured = io.StringIO()
        sys.stdout = captured
        gf.summary(dec=3)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Goodness of fit test" in output
        assert "Income" in output
        assert "Observed" in output
        assert "Expected" in output
        assert "Chi-squared" in output

    def test_summary_output_all_tables(self, newspaper_data):
        """Test summary with all output tables."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        captured = io.StringIO()
        sys.stdout = captured
        gf.summary(output=["observed", "expected", "chisq", "dev_std"], dec=3)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Observed" in output
        assert "Expected" in output
        assert "Contribution to chi-squared" in output
        assert "Deviation standardized" in output

    def test_summary_invalid_variable(self):
        """Test summary raises error for invalid variable."""
        df = pl.DataFrame({"category": ["A", "B", "C"]})
        gf = goodness(data=df, var="category")
        # Change var to invalid after init
        gf.var = "invalid"
        captured = io.StringIO()
        sys.stdout = captured
        with pytest.raises(ValueError, match="does not exist"):
            gf.summary()
        sys.stdout = sys.__stdout__


class TestGoodnessValidation:
    """Test validation and edge cases."""

    def test_goodness_prob_mismatch(self):
        """Test error when probs don't match number of levels."""
        df = pl.DataFrame({"category": ["A", "B", "C", "D"]})
        # Error occurs during init when probs don't match nlev
        with pytest.raises(IndexError):
            goodness(data=df, var="category", probs=(0.5, 0.5))  # Only 2 probs for 4 levels

    def test_goodness_prob_not_sum_one(self):
        """Test error when probs don't sum to 1."""
        df = pl.DataFrame({"category": ["A", "B", "C", "D"]})
        gf = goodness(data=df, var="category", probs=(0.1, 0.1, 0.1, 0.1))  # Sum = 0.4
        captured = io.StringIO()
        sys.stdout = captured
        with pytest.raises(ValueError, match="do not sum to 1"):
            gf.summary()
        sys.stdout = sys.__stdout__


class TestGoodnessBaseline:
    """Baseline tests to capture outputs before plotnine conversion."""

    def test_baseline_tables(self, newspaper_data):
        """Capture baseline frequency tables."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )

        # Verify observed table structure
        assert isinstance(gf.observed, pl.DataFrame)
        assert "Total" in gf.observed.columns

        # Verify expected table structure
        assert isinstance(gf.expected, pl.DataFrame)
        assert "Total" in gf.expected.columns

        # Verify chi-square contributions
        assert isinstance(gf.chisq, pl.DataFrame)
        assert gf.chisq["Total"].item() >= 0

        # Verify standardized deviations
        assert isinstance(gf.stdev, pl.DataFrame)

    def test_baseline_stats(self, newspaper_data):
        """Capture baseline chi-square test statistics."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )

        # Chi-square test results are embedded in chisq table
        total_chisq = gf.chisq["Total"].item()
        assert total_chisq >= 0

    def test_baseline_plot_observed(self, newspaper_data, baseline_plot_dir):
        """Save baseline observed plot (now using plotnine)."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["observed"])

        out_file = baseline_plot_dir / "goodness_observed_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_expected(self, newspaper_data, baseline_plot_dir):
        """Save baseline expected plot (now using plotnine)."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["expected"])

        out_file = baseline_plot_dir / "goodness_expected_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_chisq(self, newspaper_data, baseline_plot_dir):
        """Save baseline chi-square contribution plot (now using plotnine)."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["chisq"])

        out_file = baseline_plot_dir / "goodness_chisq_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_dev_std(self, newspaper_data, baseline_plot_dir):
        """Save baseline standardized deviation plot (now using plotnine)."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["dev_std"])

        out_file = baseline_plot_dir / "goodness_dev_std_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000


class TestGoodnessPlot:
    """Test plot functionality."""

    def test_plot_observed(self, newspaper_data):
        """Test observed plot generation."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["observed"])
        assert p is not None

    def test_plot_expected(self, newspaper_data):
        """Test expected plot generation."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["expected"])
        assert p is not None

    def test_plot_chisq(self, newspaper_data):
        """Test chi-square contribution plot generation."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["chisq"])
        assert p is not None

    def test_plot_dev_std(self, newspaper_data):
        """Test standardized deviation plot generation."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["dev_std"])
        assert p is not None

    def test_plot_all_types(self, newspaper_data):
        """Test all plot types together."""
        gf = goodness(
            data={"newspaper": newspaper_data},
            var="Income",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = gf.plot(plots=["observed", "expected", "chisq", "dev_std"])
        assert p is not None
