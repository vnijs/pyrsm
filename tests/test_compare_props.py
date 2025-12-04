"""Tests for pyrsm.basics.compare_props module."""

import warnings
import io
import sys
import pytest
import polars as pl
import numpy as np
from pyrsm.basics import compare_props
import pyrsm as rsm


@pytest.fixture(scope="module")
def salary_data():
    """Load the salary dataset."""
    salary, _ = rsm.load_data(pkg="basics", name="salary")
    return salary


@pytest.fixture
def synthetic_group_prop_data():
    """Create synthetic grouped proportion data for testing."""
    np.random.seed(42)
    # Create data with different proportions per group
    groups = ["A"] * 100 + ["B"] * 100 + ["C"] * 100
    # A: 70% yes, B: 50% yes, C: 30% yes
    outcomes_a = ["yes"] * 70 + ["no"] * 30
    outcomes_b = ["yes"] * 50 + ["no"] * 50
    outcomes_c = ["yes"] * 30 + ["no"] * 70
    np.random.shuffle(outcomes_a)
    np.random.shuffle(outcomes_b)
    np.random.shuffle(outcomes_c)
    outcomes = outcomes_a + outcomes_b + outcomes_c
    return pl.DataFrame({"group": groups, "outcome": outcomes})


class TestComparePropBasic:
    """Basic tests for compare_props functionality."""

    def test_compare_props_basic(self, salary_data):
        """Test basic compare_props with salary data."""
        cp = compare_props(
            data={"salary": salary_data},
            var1="rank",
            var2="sex",
            lev="Male",
            conf=0.95,
        )
        assert cp.name == "salary"
        assert cp.var1 == "rank"
        assert cp.var2 == "sex"
        assert cp.lev == "Male"
        assert len(cp.levels) > 0
        assert len(cp.descriptive_stats) > 0
        assert len(cp.comp_stats) > 0

    def test_compare_props_known_proportions(self, synthetic_group_prop_data):
        """Test with known proportions."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
        )
        # Check that descriptive stats captured the right proportions - use polars syntax
        desc = cp.descriptive_stats
        assert desc.filter(pl.col("group") == "A")["p"].item() == pytest.approx(0.7, abs=0.01)
        assert desc.filter(pl.col("group") == "B")["p"].item() == pytest.approx(0.5, abs=0.01)
        assert desc.filter(pl.col("group") == "C")["p"].item() == pytest.approx(0.3, abs=0.01)


class TestComparePropAlternatives:
    """Test alternative hypothesis options."""

    def test_compare_props_two_sided(self, synthetic_group_prop_data):
        """Test two-sided alternative."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
            alt_hyp="two-sided",
        )
        assert cp.alt_hyp == "two-sided"
        # All p-values should be valid
        assert all(0 <= p <= 1 for p in cp.comp_stats["p.value"])

    def test_compare_props_less(self, synthetic_group_prop_data):
        """Test less than alternative."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
            alt_hyp="less",
        )
        assert cp.alt_hyp == "less"

    def test_compare_props_greater(self, synthetic_group_prop_data):
        """Test greater than alternative."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
            alt_hyp="greater",
        )
        assert cp.alt_hyp == "greater"


class TestComparePropAdjustment:
    """Test p-value adjustment methods."""

    def test_compare_props_bonferroni(self, synthetic_group_prop_data):
        """Test Bonferroni adjustment."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
            adjust="bonferroni",
        )
        assert cp.adjust == "bonferroni"
        # Adjusted p-values should still be valid
        assert all(0 <= p <= 1 for p in cp.comp_stats["p.value"])

    def test_compare_props_fdr(self, synthetic_group_prop_data):
        """Test FDR adjustment."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
            adjust="fdr_bh",
        )
        assert cp.adjust == "fdr_bh"


class TestComparePropInputFormats:
    """Test different input formats."""

    def test_compare_props_polars_input(self, synthetic_group_prop_data):
        """Test with polars DataFrame input."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
        )
        assert len(cp.levels) == 3

    def test_compare_props_dict_input(self, synthetic_group_prop_data):
        """Test with dict input format."""
        cp = compare_props(
            data={"test_data": synthetic_group_prop_data},
            var1="group",
            var2="outcome",
            lev="yes",
        )
        assert cp.name == "test_data"


class TestComparePropCombinations:
    """Test specific combinations of comparisons."""

    def test_compare_props_specific_combination(self, synthetic_group_prop_data):
        """Test with specific combination of levels."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
            comb=["A:B"],
        )
        assert len(cp.comp_stats) == 1
        assert cp.comp_stats["Null hyp."].item() == "A = B"

    def test_compare_props_multiple_combinations(self, synthetic_group_prop_data):
        """Test with multiple specific combinations."""
        cp = compare_props(
            data=synthetic_group_prop_data,
            var1="group",
            var2="outcome",
            lev="yes",
            comb=["A:B", "B:C"],
        )
        assert len(cp.comp_stats) == 2


class TestComparePropSummary:
    """Test summary output."""

    def test_summary_output(self, salary_data):
        """Test that summary produces output."""
        cp = compare_props(
            data={"salary": salary_data},
            var1="rank",
            var2="sex",
            lev="Male",
        )
        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        cp.summary(dec=3)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Pairwise proportion comparisons" in output
        assert "salary" in output
        assert "Signif. codes" in output

    def test_summary_extra_output(self, salary_data):
        """Test summary with extra=True."""
        cp = compare_props(
            data={"salary": salary_data},
            var1="rank",
            var2="sex",
            lev="Male",
        )
        captured = io.StringIO()
        sys.stdout = captured
        cp.summary(extra=True, dec=3)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        # Check that extra columns are in comp_stats (df column check is more reliable than print check)
        assert "chisq.value" in cp.comp_stats.columns
        assert "df" in cp.comp_stats.columns
        # Verify output includes the basic sections
        assert "Pairwise proportion comparisons" in output


class TestComparePropPlot:
    """Test plot functionality."""

    def test_plot_bar(self, salary_data):
        """Test bar plot generation."""
        cp = compare_props(
            data={"salary": salary_data},
            var1="rank",
            var2="sex",
            lev="Male",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = cp.plot(plots="bar")
        assert p is not None

    def test_plot_dodge(self, salary_data):
        """Test dodge bar plot generation."""
        cp = compare_props(
            data={"salary": salary_data},
            var1="rank",
            var2="sex",
            lev="Male",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = cp.plot(plots="dodge")
        assert p is not None
