"""Tests for pyrsm.model.rforest module (Random Forest)."""

import io
import os
import sys
import pytest
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import pyrsm as rsm
from pyrsm.model.rforest import rforest

# Directory for saving plot comparisons
PLOT_DIR = "tests/plot_comparisons/model"
os.makedirs(PLOT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def titanic_data():
    """Load the titanic dataset for classification."""
    titanic, _ = rsm.load_data(pkg="model", name="titanic")
    return titanic.drop_nulls(subset=["age"]).to_pandas()


@pytest.fixture(scope="module")
def salary_data():
    """Load the salary dataset for regression."""
    salary, _ = rsm.load_data(pkg="basics", name="salary")
    return salary.to_pandas()


@pytest.fixture(scope="module")
def titanic_polars():
    """Load titanic as polars for testing."""
    titanic, _ = rsm.load_data(pkg="model", name="titanic")
    return titanic.drop_nulls(subset=["age"])


class TestRforestClassification:
    """Tests for random forest classification."""

    def test_rforest_classification_basic(self, titanic_data):
        """Test basic classification fitting."""
        rf = rforest(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,  # Small for speed
        )
        assert rf.name == "titanic"
        assert rf.rvar == "survived"
        assert rf.mod_type == "classification"
        assert rf.fitted is not None

    def test_rforest_classification_polars_input(self, titanic_polars):
        """Test with polars DataFrame input."""
        rf = rforest(
            data=titanic_polars,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
            n_estimators=10,
        )
        assert rf.fitted is not None

    def test_rforest_classification_oob_score(self, titanic_data):
        """Test OOB score calculation."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            oob_score=True,
            n_estimators=20,
        )
        assert hasattr(rf.fitted, "oob_score_")
        assert 0 <= rf.fitted.oob_score_ <= 1


class TestRforestRegression:
    """Tests for random forest regression."""

    def test_rforest_regression_basic(self, salary_data):
        """Test basic regression fitting."""
        rf = rforest(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        assert rf.name == "salary"
        assert rf.rvar == "salary"
        assert rf.mod_type == "regression"
        assert rf.fitted is not None

    def test_rforest_regression_oob_score(self, salary_data):
        """Test OOB R² for regression."""
        rf = rforest(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            oob_score=True,
            n_estimators=20,
        )
        assert hasattr(rf.fitted, "oob_score_")


class TestRforestSummary:
    """Tests for summary output."""

    def test_summary_classification(self, titanic_data):
        """Test summary for classification."""
        rf = rforest(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        captured = io.StringIO()
        sys.stdout = captured
        rf.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Random Forest" in output
        assert "Classification" in output or "classification" in output

    def test_summary_regression(self, salary_data):
        """Test summary for regression."""
        rf = rforest(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        captured = io.StringIO()
        sys.stdout = captured
        rf.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Random Forest" in output


class TestRforestPredict:
    """Tests for prediction functionality."""

    def test_predict_classification(self, titanic_data):
        """Test classification predictions."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        pred = rf.predict()
        assert "prediction" in pred.columns
        # Predictions should be probabilities
        assert all(pred["prediction"] >= 0)
        assert all(pred["prediction"] <= 1)

    def test_predict_regression(self, salary_data):
        """Test regression predictions."""
        rf = rforest(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        pred = rf.predict()
        assert "prediction" in pred.columns
        # Predictions should be non-negative (salary)
        assert (pred["prediction"] >= 0).all()
        # Mean prediction should be in reasonable range
        assert pred["prediction"].mean() > 50000

    def test_predict_new_data(self, titanic_data):
        """Test prediction with new data."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        new_data = titanic_data.head(10)
        pred = rf.predict(data=new_data)
        assert len(pred) == 10


class TestRforestPlot:
    """Tests for plotting functionality."""

    def test_plot_pred_classification(self, titanic_data):
        """Test prediction plot for classification."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
            n_estimators=10,
        )
        rf.plot(plots="pred", incl=["age"])
        plt.savefig(f"{PLOT_DIR}/rforest_pred_class.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_pred_regression(self, salary_data):
        """Test prediction plot for regression."""
        rf = rforest(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        rf.plot(plots="pred", incl=["yrs_since_phd"])
        plt.savefig(f"{PLOT_DIR}/rforest_pred_reg.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_vimp(self, titanic_data):
        """Test variable importance plot."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        rf.plot(plots="vimp")
        plt.savefig(f"{PLOT_DIR}/rforest_vimp.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_dashboard(self, salary_data):
        """Test dashboard plot for regression."""
        rf = rforest(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        rf.plot(plots="dashboard", nobs=100)
        plt.savefig(f"{PLOT_DIR}/rforest_dashboard.png", dpi=100, bbox_inches="tight")
        plt.close("all")


class TestRforestFeatureImportance:
    """Tests for feature importance."""

    def test_feature_importance_exists(self, titanic_data):
        """Test that feature importance is computed."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        assert hasattr(rf.fitted, "feature_importances_")
        assert len(rf.fitted.feature_importances_) > 0

    def test_feature_importance_sums_to_one(self, titanic_data):
        """Test feature importances sum to approximately 1."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        imp_sum = sum(rf.fitted.feature_importances_)
        assert 0.99 <= imp_sum <= 1.01
