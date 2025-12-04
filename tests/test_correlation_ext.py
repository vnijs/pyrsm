import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyrsm.basics.correlation import correlation


class TestCorrelationBaseline:
    """Baseline tests to capture outputs before plotnine conversion."""

    def test_baseline_correlation_matrices(self, salary_data):
        """Capture baseline correlation, p-value, and covariance matrices."""
        numeric_cols = ["salary", "yrs_since_phd", "yrs_service"]
        cr = correlation({"salary": salary_data}, vars=numeric_cols)

        # Verify structure
        assert cr.cr.shape == (3, 3)
        assert cr.cp.shape == (3, 3)
        assert cr.cv.shape == (3, 3)

        # Verify correlations are in valid range
        assert np.all(np.abs(cr.cr) <= 1.0)

        # Verify p-values are in valid range
        assert np.all((cr.cp >= 0) & (cr.cp <= 1))

        # Verify symmetry
        assert np.allclose(cr.cr, cr.cr.T)
        assert np.allclose(cr.cp, cr.cp.T)

    def test_baseline_correlation_methods(self, salary_data):
        """Test different correlation methods."""
        numeric_cols = ["salary", "yrs_since_phd", "yrs_service"]

        cr_pearson = correlation({"salary": salary_data}, vars=numeric_cols, method="pearson")
        cr_spearman = correlation({"salary": salary_data}, vars=numeric_cols, method="spearman")
        cr_kendall = correlation({"salary": salary_data}, vars=numeric_cols, method="kendall")

        # All should produce valid correlation matrices
        for cr in [cr_pearson, cr_spearman, cr_kendall]:
            assert np.all(np.abs(cr.cr) <= 1.0)
            assert np.all((cr.cp >= 0) & (cr.cp <= 1))

    def test_baseline_plot_correlation_matrix(self, salary_data, baseline_plot_dir):
        """Save baseline correlation matrix plot."""
        import matplotlib.pyplot as plt

        numeric_cols = ["salary", "yrs_since_phd", "yrs_service"]
        cr = correlation({"salary": salary_data}, vars=numeric_cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig, axes = cr.plot(nobs=200, figsize=(6, 6))

        out_file = baseline_plot_dir / "correlation_matrix.png"
        fig.savefig(out_file, dpi=100)
        plt.close("all")

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_plot_returns_fig_axes(self, salary_data):
        """Test that plot returns (Figure, Axes) tuple."""
        import matplotlib.pyplot as plt

        numeric_cols = ["salary", "yrs_since_phd", "yrs_service"]
        cr = correlation({"salary": salary_data}, vars=numeric_cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig, axes = cr.plot()

        assert fig is not None
        assert axes is not None
        plt.close("all")


def test_correlation_auto_vars():
    """Test correlation with auto-detected numeric variables."""
    df = pl.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, 1, 2, 2, 10]})
    c = correlation(df)

    assert np.isclose(c.cr[1, 0], -0.493, atol=1e-3)
    assert c.cr[0, 1] == c.cr[1, 0]
    assert c.cp[0, 1] == c.cp[1, 0]


def test_correlation_polars_pairwise_missing():
    """Test correlation with missing values."""
    df = pl.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, None, 2, 2, 10]})
    c = correlation(df)

    assert np.isclose(c.cr[1, 0], -0.4472, atol=1e-3)
    assert c.cp[1, 0] <= 1
    assert np.isfinite(c.cv).all()


def test_correlation_spearman_monotonic():
    """Test spearman correlation with monotonic data."""
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    c = correlation(df, method="spearman")

    assert np.isclose(c.cr[1, 0], 1.0)
    assert c.cp[1, 0] <= 0.05
