"""Tests for pyrsm.basics.central_limit_theorem module."""

import warnings
import pytest
import numpy as np
from pyrsm.basics import central_limit_theorem


class TestCLTNormalDistribution:
    """Test CLT with normal distribution."""

    def test_clt_normal_basic(self):
        """Test basic normal distribution simulation."""
        clt = central_limit_theorem(
            dist="normal",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            mean=0,
            sd=1,
        )
        assert clt.dist == "normal"
        assert clt.sample_size == 100
        assert clt.num_samples == 50
        assert clt.num_bins == 20
        assert clt.params["mean"] == 0
        assert clt.params["sd"] == 1

    def test_clt_normal_simulate(self):
        """Test normal distribution simulation runs and returns plots."""
        clt = central_limit_theorem(
            dist="normal",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            figsize=(8, 8),
            mean=0,
            sd=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clt.simulate()
        # simulate() calls plot_distribution internally which stores plots

    def test_clt_normal_custom_params(self):
        """Test normal distribution with custom parameters."""
        clt = central_limit_theorem(
            dist="normal",
            sample_size=200,
            num_samples=100,
            num_bins=30,
            mean=10,
            sd=5,
        )
        assert clt.params["mean"] == 10
        assert clt.params["sd"] == 5


class TestCLTBinomialDistribution:
    """Test CLT with binomial distribution."""

    def test_clt_binomial_basic(self):
        """Test basic binomial distribution simulation."""
        clt = central_limit_theorem(
            dist="binomial",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            size=10,
            prob=0.5,
        )
        assert clt.dist == "binomial"
        assert clt.params["size"] == 10
        assert clt.params["prob"] == 0.5

    def test_clt_binomial_simulate(self):
        """Test binomial distribution simulation runs."""
        clt = central_limit_theorem(
            dist="binomial",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            figsize=(8, 8),
            size=10,
            prob=0.3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clt.simulate()


class TestCLTUniformDistribution:
    """Test CLT with uniform distribution."""

    def test_clt_uniform_basic(self):
        """Test basic uniform distribution simulation."""
        clt = central_limit_theorem(
            dist="uniform",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            min=0,
            max=1,
        )
        assert clt.dist == "uniform"
        assert clt.params["min"] == 0
        assert clt.params["max"] == 1

    def test_clt_uniform_simulate(self):
        """Test uniform distribution simulation runs."""
        clt = central_limit_theorem(
            dist="uniform",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            figsize=(8, 8),
            min=0,
            max=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clt.simulate()


class TestCLTExponentialDistribution:
    """Test CLT with exponential distribution."""

    def test_clt_exponential_basic(self):
        """Test basic exponential distribution simulation."""
        clt = central_limit_theorem(
            dist="exponential",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            rate=1.0,
        )
        assert clt.dist == "exponential"
        assert clt.params["rate"] == 1.0

    def test_clt_exponential_simulate(self):
        """Test exponential distribution simulation runs."""
        clt = central_limit_theorem(
            dist="exponential",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            figsize=(8, 8),
            rate=2.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clt.simulate()


class TestCLTEdgeCases:
    """Test edge cases and validation."""

    def test_clt_invalid_distribution(self):
        """Test with invalid distribution name."""
        clt = central_limit_theorem(
            dist="invalid",
            sample_size=100,
            num_samples=50,
            num_bins=20,
        )
        # Should print "Invalid distribution" but not crash
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clt.simulate()  # Should handle gracefully

    def test_clt_bins_clamping(self):
        """Test that bins are clamped to valid range."""
        # Test num_bins > 50 gets clamped
        clt = central_limit_theorem(
            dist="normal",
            sample_size=100,
            num_samples=50,
            num_bins=100,
            mean=0,
            sd=1,
        )
        assert clt.num_bins == 50  # Should be clamped to 50

        # Test num_bins < 1 gets clamped
        clt = central_limit_theorem(
            dist="normal",
            sample_size=100,
            num_samples=50,
            num_bins=0,
            mean=0,
            sd=1,
        )
        assert clt.num_bins == 1  # Should be clamped to 1

    def test_clt_small_sample_size(self):
        """Test with small sample size."""
        clt = central_limit_theorem(
            dist="normal",
            sample_size=5,
            num_samples=10,
            num_bins=5,
            figsize=(6, 6),
            mean=0,
            sd=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clt.simulate()

    def test_clt_large_sample_size(self):
        """Test with larger sample size."""
        clt = central_limit_theorem(
            dist="normal",
            sample_size=1000,
            num_samples=100,
            num_bins=30,
            figsize=(10, 10),
            mean=0,
            sd=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clt.simulate()


class TestCLTFigsize:
    """Test figsize parameter."""

    def test_clt_custom_figsize(self):
        """Test with custom figure size."""
        clt = central_limit_theorem(
            dist="normal",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            figsize=(12, 8),
            mean=0,
            sd=1,
        )
        assert clt.figsize == (12, 8)

    def test_clt_default_figsize(self):
        """Test default figure size."""
        clt = central_limit_theorem(
            dist="normal",
            sample_size=100,
            num_samples=50,
            num_bins=20,
            mean=0,
            sd=1,
        )
        assert clt.figsize == (10, 10)
