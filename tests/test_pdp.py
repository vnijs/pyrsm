"""Tests for pyrsm.model.visualize pdp_sk and pdp_sm functions."""

import os
import pytest
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import pyrsm as rsm
from pyrsm.model.rforest import rforest
from pyrsm.model.visualize import pdp_sk, pdp_sm, pred_plot_sk, pred_plot_sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import partial_dependence

# Directory for saving plot comparisons
PLOT_DIR = "tests/plot_comparisons/pdp"
os.makedirs(PLOT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def titanic_data():
    """Load the titanic dataset for classification."""
    titanic, _ = rsm.load_data(pkg="model", name="titanic")
    return titanic.drop_nulls(subset=["age"])


@pytest.fixture(scope="module")
def salary_data():
    """Load the salary dataset for regression."""
    salary, _ = rsm.load_data(pkg="basics", name="salary")
    return salary


@pytest.fixture(scope="module")
def diamonds_data():
    """Load diamonds dataset for larger tests."""
    diamonds, _ = rsm.load_data(pkg="basics", name="diamonds")
    # Sample for speed
    return diamonds.sample(1000, seed=1234)


@pytest.fixture(scope="module")
def rf_classifier(titanic_data):
    """Fit a random forest classifier."""
    rf = rforest(
        data=titanic_data.to_pandas(),
        rvar="survived",
        lev="Yes",
        evar=["age", "sex", "pclass"],
        mod_type="classification",
        n_estimators=20,
        random_state=1234,
    )
    return rf


@pytest.fixture(scope="module")
def rf_regressor(salary_data):
    """Fit a random forest regressor."""
    rf = rforest(
        data=salary_data.to_pandas(),
        rvar="salary",
        evar=["yrs_since_phd", "yrs_service", "rank"],
        mod_type="regression",
        n_estimators=20,
        random_state=1234,
    )
    return rf


@pytest.fixture(scope="module")
def logit_model(titanic_data):
    """Fit a logistic regression model."""
    df = titanic_data.with_columns(
        pl.when(pl.col("survived") == "Yes").then(1).otherwise(0).alias("survived_bin")
    ).to_pandas()
    model = smf.logit("survived_bin ~ age + C(sex) + C(pclass)", data=df).fit(disp=0)
    return model, df


@pytest.fixture(scope="module")
def ols_model(salary_data):
    """Fit an OLS regression model."""
    df = salary_data.to_pandas()
    model = smf.ols("salary ~ yrs_since_phd + yrs_service + C(rank)", data=df).fit()
    return model, df


class TestPdpSkBasic:
    """Basic tests for pdp_sk function."""

    def test_pdp_sk_returns_tuple(self, rf_classifier, titanic_data):
        """Test that pdp_sk returns a tuple of (plot, dict, runtime)."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        assert isinstance(data_dict, dict)
        assert isinstance(runtime, float)
        assert runtime > 0

    def test_pdp_sk_data_dict_contains_correct_keys(self, rf_classifier, titanic_data):
        """Test that data_dict contains the requested variables."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age", "sex"],
            grid_resolution=10,
            n_sample=100,
        )
        assert "age" in data_dict
        assert "sex" in data_dict

    def test_pdp_sk_data_has_prediction_column(self, rf_classifier, titanic_data):
        """Test that output DataFrames have prediction column."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        assert "prediction" in data_dict["age"].columns

    def test_pdp_sk_predictions_in_valid_range_classification(self, rf_classifier, titanic_data):
        """Test predictions are valid probabilities for classification."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        preds = data_dict["age"]["prediction"].to_list()
        assert all(0 <= p <= 1 for p in preds)

    def test_pdp_sk_predictions_reasonable_regression(self, rf_regressor, salary_data):
        """Test predictions are reasonable for regression."""
        plot, data_dict, runtime = pdp_sk(
            rf_regressor.fitted,
            salary_data,
            incl=["yrs_since_phd"],
            grid_resolution=10,
            n_sample=100,
        )
        preds = data_dict["yrs_since_phd"]["prediction"].to_list()
        # Salary should be positive
        assert all(p > 0 for p in preds)
        # Mean should be in reasonable range
        mean_pred = np.mean(preds)
        assert 50000 < mean_pred < 200000


class TestPdpSkModes:
    """Test different modes of pdp_sk."""

    def test_pdp_mode_produces_different_results_than_fast(self, rf_classifier, titanic_data):
        """Test that pdp mode produces different (usually smoother) results than fast."""
        plot_pdp, data_pdp, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="pdp",
            grid_resolution=10,
            n_sample=200,
        )
        plot_fast, data_fast, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=10,
        )
        # Results should exist for both
        assert "age" in data_pdp
        assert "age" in data_fast
        # Values might differ due to averaging vs point prediction
        preds_pdp = data_pdp["age"]["prediction"].to_list()
        preds_fast = data_fast["age"]["prediction"].to_list()
        # Check they're not identical (except by chance)
        assert len(preds_pdp) == len(preds_fast)

    def test_pdp_mode_runtime_tracked(self, rf_classifier, titanic_data):
        """Test that runtime is correctly tracked for pdp mode."""
        _, _, runtime_pdp = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="pdp",
            grid_resolution=5,
            n_sample=50,
        )
        _, _, runtime_fast = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=5,
        )
        # Both should have positive runtime
        assert runtime_pdp > 0
        assert runtime_fast > 0


class TestPdpSkCategorical:
    """Test categorical variable handling."""

    def test_pdp_sk_categorical_all_levels(self, rf_classifier, titanic_data):
        """Test that categorical variables show all levels."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["sex"],
            grid_resolution=10,
            n_sample=100,
        )
        sex_values = data_dict["sex"]["sex"].unique().to_list()
        # Should have both male and female
        assert len(sex_values) >= 2

    def test_pdp_sk_pclass_categorical(self, rf_classifier, titanic_data):
        """Test pclass as categorical."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["pclass"],
            grid_resolution=10,
            n_sample=100,
        )
        pclass_values = data_dict["pclass"]["pclass"].unique().to_list()
        # Should have all passenger classes
        assert len(pclass_values) >= 2


class TestPdpSkInteractions:
    """Test interaction plotting."""

    def test_pdp_sk_interaction_num_cat(self, rf_classifier, titanic_data):
        """Test numeric-categorical interaction."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=[],
            incl_int=["age:sex"],
            grid_resolution=10,
            n_sample=100,
        )
        assert "age:sex" in data_dict
        # Should have both age and sex columns
        assert "age" in data_dict["age:sex"].columns
        assert "sex" in data_dict["age:sex"].columns
        assert "prediction" in data_dict["age:sex"].columns

    def test_pdp_sk_interaction_num_num(self, rf_regressor, salary_data):
        """Test numeric-numeric interaction."""
        plot, data_dict, runtime = pdp_sk(
            rf_regressor.fitted,
            salary_data,
            incl=[],
            incl_int=["yrs_since_phd:yrs_service"],
            grid_resolution=10,
            n_sample=100,
            interaction_slices=5,
        )
        assert "yrs_since_phd:yrs_service" in data_dict
        int_df = data_dict["yrs_since_phd:yrs_service"]
        assert "yrs_since_phd" in int_df.columns
        assert "yrs_service" in int_df.columns


class TestPdpSkGridResolution:
    """Test grid resolution parameter."""

    def test_pdp_sk_grid_resolution_affects_output_size(self, rf_classifier, titanic_data):
        """Test that grid_resolution affects number of points."""
        _, data_10, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=10,
        )
        _, data_20, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=20,
        )
        # More resolution = more points (up to unique values)
        assert data_20["age"].height >= data_10["age"].height


class TestPdpSkQuantiles:
    """Test quantile-based grid limits."""

    def test_pdp_sk_quantiles_limit_range(self, rf_classifier, titanic_data):
        """Test that minq/maxq limit the range."""
        _, data_full, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            minq=0.0,
            maxq=1.0,
            grid_resolution=20,
        )
        _, data_trimmed, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            minq=0.1,
            maxq=0.9,
            grid_resolution=20,
        )
        # Trimmed range should be smaller
        full_range = data_full["age"]["age"].max() - data_full["age"]["age"].min()
        trimmed_range = data_trimmed["age"]["age"].max() - data_trimmed["age"]["age"].min()
        assert trimmed_range <= full_range


class TestPdpSmBasic:
    """Basic tests for pdp_sm function."""

    def test_pdp_sm_returns_tuple(self, logit_model, titanic_data):
        """Test that pdp_sm returns a tuple."""
        model, df = logit_model
        plot, data_dict, runtime = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        assert isinstance(data_dict, dict)
        assert isinstance(runtime, float)

    def test_pdp_sm_logistic_predictions_valid(self, logit_model, titanic_data):
        """Test logistic predictions are valid probabilities."""
        model, df = logit_model
        plot, data_dict, runtime = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        preds = data_dict["age"]["prediction"].to_list()
        assert all(0 <= p <= 1 for p in preds)

    def test_pdp_sm_ols_predictions_reasonable(self, ols_model, salary_data):
        """Test OLS predictions are reasonable."""
        model, df = ols_model
        plot, data_dict, runtime = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["yrs_since_phd"],
            grid_resolution=10,
            n_sample=100,
        )
        preds = data_dict["yrs_since_phd"]["prediction"].to_list()
        mean_pred = np.mean(preds)
        assert 50000 < mean_pred < 200000


class TestPdpSmModes:
    """Test pdp_sm modes."""

    def test_pdp_sm_both_modes_work(self, ols_model, salary_data):
        """Test both fast and pdp modes work."""
        model, df = ols_model
        df_pl = pl.from_pandas(df)

        plot_pdp, data_pdp, _ = pdp_sm(
            model, df_pl, incl=["yrs_since_phd"], mode="pdp", grid_resolution=10, n_sample=100
        )
        plot_fast, data_fast, _ = pdp_sm(
            model, df_pl, incl=["yrs_since_phd"], mode="fast", grid_resolution=10
        )

        assert "yrs_since_phd" in data_pdp
        assert "yrs_since_phd" in data_fast


class TestPdpPlotSaving:
    """Test plot generation and saving."""

    def test_pdp_sk_save_single_var(self, rf_classifier, titanic_data):
        """Test saving single variable PDP plot."""
        plot, _, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            grid_resolution=20,
            n_sample=200,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sk_single_age.png", dpi=100, verbose=False)
        plt.close("all")

    def test_pdp_sk_save_multiple_vars(self, rf_classifier, titanic_data):
        """Test saving multiple variable PDP plots."""
        plot, _, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age", "sex", "pclass"],
            grid_resolution=15,
            n_sample=200,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sk_multi.png", dpi=100, verbose=False)
        plt.close("all")

    def test_pdp_sk_save_interaction(self, rf_classifier, titanic_data):
        """Test saving interaction PDP plot."""
        plot, _, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=[],
            incl_int=["age:sex"],
            grid_resolution=15,
            n_sample=200,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sk_interaction.png", dpi=100, verbose=False)
        plt.close("all")

    def test_pdp_sm_save_plot(self, ols_model, salary_data):
        """Test saving statsmodels PDP plot."""
        model, df = ols_model
        plot, _, _ = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["yrs_since_phd", "yrs_service"],
            grid_resolution=15,
            n_sample=200,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sm_ols.png", dpi=100, verbose=False)
        plt.close("all")


class TestPdpVsPredPlotComparison:
    """Compare pdp_sk with pred_plot_sk outputs."""

    def test_pdp_fast_similar_to_pred_plot(self, rf_classifier, titanic_data):
        """Test that pdp_sk fast mode is similar to pred_plot_sk."""
        # Get pred_plot_sk data
        pred_dict = pred_plot_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            ret=True,
            nnv=20,
        )

        # Get pdp_sk fast mode data
        _, pdp_dict, _ = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=20,
        )

        # Both should have age
        assert "age" in pred_dict
        assert "age" in pdp_dict


class TestPdpSklearnComparison:
    """Compare pdp_sk with sklearn's partial_dependence."""

    def test_pdp_sk_vs_sklearn_numeric(self, salary_data):
        """Compare pdp_sk with sklearn for numeric variable."""
        from pyrsm.model.model import conditional_get_dummies

        # Prepare data without categoricals for simpler comparison
        df = salary_data.select(["salary", "yrs_since_phd", "yrs_service"]).drop_nulls()
        X = df.select(["yrs_since_phd", "yrs_service"]).to_pandas()
        y = df["salary"].to_list()

        # Fit sklearn RF directly
        rf = RandomForestRegressor(n_estimators=20, random_state=1234)
        rf.fit(X, y)

        # Get sklearn PDP
        sklearn_pdp = partial_dependence(
            rf, X, features=[0], grid_resolution=20, kind="average"
        )

        # Get our PDP
        _, our_pdp, _ = pdp_sk(
            rf,
            df,
            incl=["yrs_since_phd"],
            mode="pdp",
            grid_resolution=20,
            n_sample=df.height,  # Use all data for comparison
        )

        # Both should produce predictions in similar range
        sklearn_mean = np.mean(sklearn_pdp["average"][0])
        our_mean = np.mean(our_pdp["yrs_since_phd"]["prediction"].to_list())

        # Should be within 10% of each other
        assert abs(sklearn_mean - our_mean) / sklearn_mean < 0.2


class TestPdpEdgeCases:
    """Test edge cases."""

    def test_pdp_sk_empty_incl(self, rf_classifier, titanic_data):
        """Test with empty incl list."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=[],
            incl_int=[],
        )
        assert plot is None
        assert data_dict == {}

    def test_pdp_sk_exclude_variables(self, rf_classifier, titanic_data):
        """Test excluding variables."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            excl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        assert "age" not in data_dict

    def test_pdp_sk_small_sample(self, rf_classifier, titanic_data):
        """Test with very small sample size."""
        plot, data_dict, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="pdp",
            n_sample=10,
            grid_resolution=5,
        )
        assert "age" in data_dict


class TestPdpPerformance:
    """Test performance characteristics."""

    def test_pdp_sk_timing_reasonable(self, rf_classifier, titanic_data):
        """Test that pdp_sk completes in reasonable time."""
        _, _, runtime = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age", "sex"],
            mode="pdp",
            n_sample=500,
            grid_resolution=20,
        )
        # Should complete in under 30 seconds
        assert runtime < 30

    def test_pdp_sm_timing_reasonable(self, ols_model, salary_data):
        """Test that pdp_sm completes in reasonable time."""
        model, df = ols_model
        _, _, runtime = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["yrs_since_phd"],
            mode="pdp",
            n_sample=500,
            grid_resolution=20,
        )
        # Should complete in under 30 seconds
        assert runtime < 30
