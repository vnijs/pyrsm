"""Tests for pyrsm.model.xgboost module (XGBoost)."""

import io
import os
import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyrsm as rsm
from pyrsm.model.xgboost import xgboost

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


class TestXGBoostClassification:
    """Tests for XGBoost classification."""

    def test_xgboost_classification_basic(self, titanic_data):
        """Test basic classification fitting."""
        xgb = xgboost(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,  # Small for speed
        )
        assert xgb.name == "titanic"
        assert xgb.rvar == "survived"
        assert xgb.mod_type == "classification"
        assert xgb.fitted is not None

    def test_xgboost_classification_polars_input(self):
        """Test with polars DataFrame input."""
        titanic, _ = rsm.load_data(pkg="model", name="titanic")
        titanic = titanic.drop_nulls(subset=["age"])
        xgb = xgboost(
            data=titanic,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
            n_estimators=10,
        )
        assert xgb.fitted is not None


class TestXGBoostRegression:
    """Tests for XGBoost regression."""

    def test_xgboost_regression_basic(self, salary_data):
        """Test basic regression fitting."""
        xgb = xgboost(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        assert xgb.name == "salary"
        assert xgb.rvar == "salary"
        assert xgb.mod_type == "regression"
        assert xgb.fitted is not None


class TestXGBoostSummary:
    """Tests for summary output."""

    def test_summary_classification(self, titanic_data):
        """Test summary for classification."""
        xgb = xgboost(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        captured = io.StringIO()
        sys.stdout = captured
        xgb.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "XGBoost" in output

    def test_summary_regression(self, salary_data):
        """Test summary for regression."""
        xgb = xgboost(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        captured = io.StringIO()
        sys.stdout = captured
        xgb.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "XGBoost" in output


class TestXGBoostPredict:
    """Tests for prediction functionality."""

    def test_predict_classification(self, titanic_data):
        """Test classification predictions."""
        xgb = xgboost(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        pred = xgb.predict()
        assert "prediction" in pred.columns
        # Predictions should be probabilities
        assert (pred["prediction"] >= 0).all()
        assert (pred["prediction"] <= 1).all()

    def test_predict_regression(self, salary_data):
        """Test regression predictions."""
        xgb = xgboost(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        pred = xgb.predict()
        assert "prediction" in pred.columns
        # Mean prediction should be reasonable
        assert pred["prediction"].mean() > 50000

    def test_predict_new_data(self, titanic_data):
        """Test prediction with new data."""
        xgb = xgboost(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        new_data = titanic_data.head(10)
        pred = xgb.predict(data=new_data)
        assert len(pred) == 10


class TestXGBoostPlot:
    """Tests for plotting functionality."""

    def test_plot_pred_classification(self, titanic_data):
        """Test prediction plot for classification."""
        xgb = xgboost(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
            n_estimators=10,
        )
        xgb.plot(plots="pred", incl=["age"])
        plt.savefig(f"{PLOT_DIR}/xgboost_pred.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_vimp(self, titanic_data):
        """Test variable importance plot."""
        xgb = xgboost(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        xgb.plot(plots="vimp")
        plt.savefig(f"{PLOT_DIR}/xgboost_vimp.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_dashboard_regression(self, salary_data):
        """Test dashboard plot for regression."""
        xgb = xgboost(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        xgb.plot(plots="dashboard", nobs=100)
        plt.savefig(f"{PLOT_DIR}/xgboost_dashboard.png", dpi=100, bbox_inches="tight")
        plt.close("all")


class TestXGBoostFeatureImportance:
    """Tests for feature importance."""

    def test_feature_importance_exists(self, titanic_data):
        """Test that feature importance is computed."""
        xgb = xgboost(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        assert hasattr(xgb.fitted, "feature_importances_")
        assert len(xgb.fitted.feature_importances_) > 0
