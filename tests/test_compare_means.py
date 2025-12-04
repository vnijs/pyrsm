import warnings
from pathlib import Path

import pytest
from matplotlib import pyplot as plt
import polars as pl

from pyrsm.basics.compare_means import compare_means
from pyrsm.basics.utils import ci_label


class TestCompareMeansBaseline:
    """Baseline tests to capture outputs before plotnine conversion."""

    def test_baseline_descriptive_stats(self, salary_data):
        """Capture baseline descriptive statistics."""
        cm = compare_means({"salary": salary_data}, var1="rank", var2="salary")

        # Verify structure
        assert isinstance(cm.descriptive_stats, pl.DataFrame)
        assert {"rank", "mean", "n", "n_missing", "sd", "se", "me"}.issubset(
            cm.descriptive_stats.columns
        )

        # Verify values are reasonable
        assert cm.descriptive_stats["n"].sum() > 0
        assert cm.descriptive_stats["mean"].is_not_null().all()
        assert cm.descriptive_stats["sd"].is_not_null().all()

    def test_baseline_comparison_stats(self, salary_data):
        """Capture baseline comparison statistics."""
        cm = compare_means({"salary": salary_data}, var1="rank", var2="salary")

        # Verify structure
        assert isinstance(cm.comp_stats, pl.DataFrame)
        assert {"Null hyp.", "Alt. hyp.", "diff", "p.value", "t.value", "df"}.issubset(
            cm.comp_stats.columns
        )

        # Verify p-values are valid
        assert cm.comp_stats["p.value"].is_between(0, 1, closed="both").all()

        # Verify t-values are present
        assert cm.comp_stats["t.value"].is_not_null().all()

    def test_baseline_comparison_stats_adjusted(self, salary_data):
        """Capture baseline with Bonferroni adjustment."""
        cm_adj = compare_means(
            {"salary": salary_data}, var1="rank", var2="salary", adjust="bonferroni"
        )
        cm_raw = compare_means({"salary": salary_data}, var1="rank", var2="salary")

        # Adjusted p-values should be >= raw p-values
        assert (cm_adj.comp_stats["p.value"] >= cm_raw.comp_stats["p.value"]).all()

    def test_baseline_plot_scatter(self, salary_data, baseline_plot_dir):
        """Save baseline scatter plot (now using plotnine)."""
        cm = compare_means({"salary": salary_data}, var1="rank", var2="salary")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = cm.plot(plots="scatter", nobs=100)

        out_file = baseline_plot_dir / "compare_means_scatter_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_box(self, salary_data, baseline_plot_dir):
        """Save baseline box plot (now using plotnine)."""
        cm = compare_means({"salary": salary_data}, var1="rank", var2="salary")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = cm.plot(plots="box")

        out_file = baseline_plot_dir / "compare_means_box_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_density(self, salary_data, baseline_plot_dir):
        """Save baseline density plot (now using plotnine)."""
        cm = compare_means({"salary": salary_data}, var1="rank", var2="salary")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = cm.plot(plots="density")

        out_file = baseline_plot_dir / "compare_means_density_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_baseline_plot_bar(self, salary_data, baseline_plot_dir):
        """Save baseline bar plot (now using plotnine)."""
        cm = compare_means({"salary": salary_data}, var1="rank", var2="salary")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = cm.plot(plots="bar")

        out_file = baseline_plot_dir / "compare_means_bar_plotnine.png"
        p.save(str(out_file), dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000


def test_compare_means_handles_basics_data(load_basics_dataset):
    df = load_basics_dataset("salary")
    cm = compare_means({"salary": df}, var1="rank", var2="salary", alt_hyp="less")

    # String cache disabled in conftest.py, so direct comparison works
    expected_levels = set(df.get_column("rank").unique().to_list())

    assert cm.name == "salary"
    assert set(cm.levels) == expected_levels
    assert {"mean", "n", "n_missing", "sd", "se", "me"}.issubset(cm.descriptive_stats.columns)

    ci_cols = ci_label("less", 0.95)
    assert list(cm.comp_stats.columns[7:9]) == ci_cols
    assert cm.comp_stats["p.value"].is_between(0, 1, closed="both").all()
    assert isinstance(cm.descriptive_stats, pl.DataFrame)
    assert isinstance(cm.comp_stats, pl.DataFrame)


def test_compare_means_numeric_var1_melts(numeric_var1_frame):
    df = numeric_var1_frame[1]
    cm = compare_means(df, var1="measurement", var2=["score_a", "score_b"])

    assert cm.var1 == "variable"
    assert cm.var2 == "value"
    assert {"measurement", "score_a", "score_b"}.issubset(set(cm.levels))
    assert cm.data.columns == ["variable", "value"]
    assert cm.data.select(pl.col("variable")).dtypes[0] == pl.Categorical


def test_compare_means_missing_values_and_zero_variance(synthetic_group_frame):
    df = synthetic_group_frame[1]
    # Suppress expected warnings from zero-variance groups and edge cases
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cm = compare_means(df, var1="group", var2="value")
    stats_df = cm.descriptive_stats.with_columns(pl.col("group"))

    stats_map = {row["group"]: row for row in stats_df.to_dicts()}
    assert stats_map["a"]["n_missing"] == 1
    assert stats_map["b"]["sd"] == 0
    assert stats_map["c"]["n_missing"] == 1
    assert set(cm.comb) == {"a:b", "a:c", "b:c"}
    assert cm.comp_stats["diff"].is_not_null().all()


def test_compare_means_adjustment_and_combination():
    df = pl.DataFrame(
        {
            "grp": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
            "value": ([0.0] * 30) + ([0.5] * 30) + ([1.0] * 30),
        }
    )
    # Suppress expected warnings from identical values within groups
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cm = compare_means(df, var1="grp", var2="value")
        cm_adj = compare_means(df, var1="grp", var2="value", adjust="bonferroni")

    assert (cm_adj.comp_stats["p.value"] >= cm.comp_stats["p.value"]).all()
    assert set(cm_adj.comb) == set(cm.comb)
    assert "***" in set(cm_adj.comp_stats[""])


def test_compare_means_wilcox_and_summary_columns(load_basics_dataset):
    df = load_basics_dataset("salary")
    cm = compare_means(
        df,
        var1="sex",
        var2="salary",
        test_type="wilcox",
        alt_hyp="greater",
        conf=0.9,
    )

    ci_cols = ci_label("greater", 0.9)
    assert list(cm.comp_stats.columns[7:9]) == ci_cols
    assert cm.comp_stats.select(["Null hyp.", "Alt. hyp.", "diff", "p.value", ""]).width == 5
    assert cm.sample_type == "independent"


def test_compare_means_raises_for_paired_size_mismatch():
    df = pl.DataFrame({"group": ["a", "a", "a", "b", "b", "b", "b", "b"], "value": range(1, 9)})

    with pytest.raises(ValueError):
        compare_means(df, var1="group", var2="value", sample_type="paired")


def test_compare_means_plot_outputs(load_basics_dataset, basics_plot_dir):
    df = load_basics_dataset("salary")
    cm = compare_means(df, var1="rank", var2="salary")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for plot in ["scatter", "box", "density", "bar"]:
            p = cm.plot(plots=plot, nobs=50)
            out_file = basics_plot_dir / f"{plot}.png"
            p.save(str(out_file), verbose=False)
            assert out_file.exists()
            assert out_file.stat().st_size > 0
