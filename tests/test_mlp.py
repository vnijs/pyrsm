"""Tests for pyrsm.model.mlp module (Multi-Layer Perceptron / Neural Network)."""

import io
import os
import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyrsm as rsm
from pyrsm.model.mlp import mlp

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


class TestMLPClassification:
    """Tests for MLP classification."""

    def test_mlp_classification_basic(self, titanic_data):
        """Test basic classification fitting."""
        nn = mlp(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            hidden_layer_sizes=(10,),  # Small for speed
            max_iter=100,
        )
        assert nn.name == "titanic"
        assert nn.rvar == "survived"
        assert nn.fitted is not None

    def test_mlp_classification_multiple_layers(self, titanic_data):
        """Test with multiple hidden layers."""
        nn = mlp(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            hidden_layer_sizes=(10, 5),
            max_iter=100,
        )
        assert nn.fitted is not None


class TestMLPRegression:
    """Tests for MLP regression."""

    def test_mlp_regression_basic(self, salary_data):
        """Test basic regression fitting."""
        nn = mlp(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(10,),
            max_iter=100,
        )
        assert nn.name == "salary"
        assert nn.rvar == "salary"
        assert nn.fitted is not None


class TestMLPSummary:
    """Tests for summary output."""

    def test_summary_classification(self, titanic_data):
        """Test summary for classification."""
        nn = mlp(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            hidden_layer_sizes=(10,),
            max_iter=100,
        )
        captured = io.StringIO()
        sys.stdout = captured
        nn.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Multi-layer Perceptron" in output or "NN" in output

    def test_summary_regression(self, salary_data):
        """Test summary for regression."""
        nn = mlp(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(10,),
            max_iter=100,
        )
        captured = io.StringIO()
        sys.stdout = captured
        nn.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Multi-layer Perceptron" in output or "NN" in output


class TestMLPPredict:
    """Tests for prediction functionality."""

    def test_predict_classification(self, titanic_data):
        """Test classification predictions."""
        nn = mlp(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            hidden_layer_sizes=(10,),
            max_iter=100,
        )
        pred = nn.predict()
        assert "prediction" in pred.columns
        # Predictions should be probabilities
        assert (pred["prediction"] >= 0).all()
        assert (pred["prediction"] <= 1).all()

    def test_predict_regression(self, salary_data):
        """Test regression predictions."""
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(10,),
            max_iter=100,
        )
        pred = nn.predict()
        assert "prediction" in pred.columns

    def test_predict_new_data(self, titanic_data):
        """Test prediction with new data."""
        nn = mlp(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            hidden_layer_sizes=(10,),
            max_iter=100,
        )
        new_data = titanic_data.head(10)
        pred = nn.predict(data=new_data)
        assert len(pred) == 10


class TestMLPPlot:
    """Tests for plotting functionality."""

    def test_plot_pred(self, titanic_data):
        """Test prediction plot."""
        nn = mlp(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            hidden_layer_sizes=(10,),
            max_iter=100,
        )
        nn.plot(plots="pred", incl=["age"])
        plt.savefig(f"{PLOT_DIR}/mlp_pred.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_vimp(self, titanic_data):
        """Test variable importance plot."""
        nn = mlp(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            hidden_layer_sizes=(10,),
            max_iter=100,
        )
        nn.plot(plots="vimp")
        plt.savefig(f"{PLOT_DIR}/mlp_vimp.png", dpi=100, bbox_inches="tight")
        plt.close("all")
