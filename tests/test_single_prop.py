"""Tests for pyrsm.basics.single_prop module."""

import warnings
import io
import sys
import pytest
import polars as pl
import numpy as np
from pyrsm.basics import single_prop
import pyrsm as rsm


@pytest.fixture(scope="module")
def consider_data():
    """Load the consider dataset."""
    consider, _ = rsm.load_data(pkg="basics", name="consider")
    return consider


@pytest.fixture
def synthetic_binary_data():
    """Create synthetic binary data for testing."""
    np.random.seed(42)
    n = 200
    # Create data with known proportions
    values = np.array(["yes"] * 60 + ["no"] * 140)
    np.random.shuffle(values)
    return pl.DataFrame({"response": values})


class TestSinglePropBasic:
    """Basic tests for single_prop functionality."""

    def test_single_prop_binomial_test(self, consider_data):
        """Test binomial test with consider data."""
        sp = single_prop(
            data={"consider": consider_data},
            var="consider",
            lev="yes",
            alt_hyp="two-sided",
            conf=0.95,
            comp_value=0.5,
            test_type="binomial",
        )
        assert sp.name == "consider"
        assert sp.var == "consider"
        assert sp.lev == "yes"
        assert sp.test_type == "binomial"
        assert 0 <= sp.p <= 1
        assert sp.n > 0
        assert 0 <= sp.p_val <= 1

    def test_single_prop_ztest(self, consider_data):
        """Test z-test with consider data."""
        sp = single_prop(
            data={"consider": consider_data},
            var="consider",
            lev="yes",
            alt_hyp="two-sided",
            conf=0.95,
            comp_value=0.5,
            test_type="z-test",
        )
        assert sp.test_type == "z-test"
        assert sp.z_score is not None
        assert 0 <= sp.p_val <= 1

    def test_single_prop_alternative_hypotheses(self, synthetic_binary_data):
        """Test different alternative hypotheses."""
        for alt in ["two-sided", "less", "greater"]:
            sp = single_prop(
                data=synthetic_binary_data,
                var="response",
                lev="yes",
                alt_hyp=alt,
                comp_value=0.5,
                test_type="binomial",
            )
            assert sp.alt_hyp == alt
            assert 0 <= sp.p_val <= 1
            # CI should be valid
            assert sp.ci[0] <= sp.ci[1]

    def test_single_prop_known_proportion(self, synthetic_binary_data):
        """Test with known proportion (60/200 = 0.3)."""
        sp = single_prop(
            data=synthetic_binary_data,
            var="response",
            lev="yes",
            comp_value=0.5,
            test_type="binomial",
        )
        # The synthetic data has 60 "yes" out of 200 = 0.3
        assert sp.p == pytest.approx(0.3, abs=0.01)
        assert sp.ns == 60
        assert sp.n == 200


class TestSinglePropInputFormats:
    """Test different input formats."""

    def test_single_prop_polars_input(self, synthetic_binary_data):
        """Test with polars DataFrame input."""
        sp = single_prop(
            data=synthetic_binary_data,
            var="response",
            lev="yes",
            comp_value=0.5,
        )
        assert sp.n == 200

    def test_single_prop_dict_input(self, synthetic_binary_data):
        """Test with dict input format."""
        sp = single_prop(
            data={"test_data": synthetic_binary_data},
            var="response",
            lev="yes",
            comp_value=0.5,
        )
        assert sp.name == "test_data"
        assert sp.n == 200


class TestSinglePropEdgeCases:
    """Test edge cases and validation."""

    def test_single_prop_invalid_comp_value_zero(self, synthetic_binary_data):
        """Test that comp_value=0 raises exception."""
        with pytest.raises(Exception, match="comparison value between 0 and 1"):
            single_prop(
                data=synthetic_binary_data,
                var="response",
                lev="yes",
                comp_value=0,
            )

    def test_single_prop_invalid_comp_value_one(self, synthetic_binary_data):
        """Test that comp_value=1 raises exception."""
        with pytest.raises(Exception, match="comparison value between 0 and 1"):
            single_prop(
                data=synthetic_binary_data,
                var="response",
                lev="yes",
                comp_value=1,
            )

    def test_single_prop_with_missing_values(self):
        """Test handling of missing values."""
        df = pl.DataFrame({
            "response": ["yes", "no", "yes", None, "no", "yes", None, "no"]
        })
        sp = single_prop(
            data=df,
            var="response",
            lev="yes",
            comp_value=0.5,
        )
        assert sp.n_missing == 2
        assert sp.n == 6  # 8 - 2 missing


class TestSinglePropSummary:
    """Test summary output."""

    def test_summary_output(self, consider_data):
        """Test that summary produces output."""
        sp = single_prop(
            data={"consider": consider_data},
            var="consider",
            lev="yes",
            comp_value=0.5,
        )
        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        sp.summary(dec=3)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Single proportion" in output
        assert "consider" in output
        assert "yes" in output
        assert "Signif. codes" in output


class TestSinglePropPlot:
    """Test plot functionality."""

    def test_plot_bar(self, consider_data):
        """Test bar plot generation."""
        sp = single_prop(
            data={"consider": consider_data},
            var="consider",
            lev="yes",
            comp_value=0.5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = sp.plot(plots="bar")
        assert p is not None
