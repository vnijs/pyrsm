"""Tests for pyrsm.model.logistic module (Logistic Regression).

Titanic dataset columns: pclass, survived, sex, age, sibsp, parch, fare, name, cabin, embarked
"""

import io
import os
import sys
import pytest
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import pyrsm as rsm
from pyrsm.model.logistic import logistic

# Directory for saving plot comparisons
PLOT_DIR = "tests/plot_comparisons/model"
os.makedirs(PLOT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def titanic_data():
    """Load the titanic dataset and convert to pandas for statsmodels."""
    titanic, _ = rsm.load_data(pkg="model", name="titanic")
    # Drop rows with missing age for simpler tests
    titanic = titanic.drop_nulls(subset=["age"])
    return titanic.to_pandas()


@pytest.fixture(scope="module")
def titanic_polars():
    """Load titanic as polars for testing polars input."""
    titanic, _ = rsm.load_data(pkg="model", name="titanic")
    return titanic.drop_nulls(subset=["age"])


class TestLogisticBasic:
    """Basic tests for logistic class initialization."""

    def test_logistic_basic_fit(self, titanic_data):
        """Test basic logistic regression fitting."""
        lr = logistic(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
        )
        assert lr.name == "titanic"
        assert lr.rvar == "survived"
        assert lr.lev == "Yes"
        assert "age" in lr.evar
        assert lr.fitted is not None
        assert hasattr(lr, "coef")

    def test_logistic_polars_input(self, titanic_polars):
        """Test with polars DataFrame input."""
        lr = logistic(
            data=titanic_polars,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        assert lr.fitted is not None

    def test_logistic_pandas_input(self, titanic_data):
        """Test with pandas DataFrame input."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age"],
        )
        assert lr.fitted is not None

    def test_logistic_dict_input(self, titanic_data):
        """Test with dict input format."""
        lr = logistic(
            data={"test_titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age"],
        )
        assert lr.name == "test_titanic"

    def test_logistic_formula(self, titanic_data):
        """Test logistic with formula input."""
        lr = logistic(
            data=titanic_data,
            form="survived ~ age + sex",
            lev="Yes",
        )
        assert lr.rvar == "survived"
        assert "age" in lr.evar
        assert "sex" in lr.evar

    def test_logistic_intercept_only(self, titanic_data):
        """Test intercept-only model."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=None,
        )
        assert "1" in lr.form


class TestLogisticCoefficients:
    """Test coefficient output."""

    def test_coef_structure(self, titanic_data):
        """Test coefficient DataFrame structure."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        assert "OR" in lr.coef.columns
        assert "OR%" in lr.coef.columns
        assert "coefficient" in lr.coef.columns
        assert "std.error" in lr.coef.columns
        assert "z.value" in lr.coef.columns
        assert "p.value" in lr.coef.columns

    def test_coef_values_reasonable(self, titanic_data):
        """Test that coefficients are reasonable."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        # Intercept should exist
        assert "Intercept" in lr.coef["index"].to_list()
        # Odds ratios should be positive
        assert (lr.coef["OR"] > 0).all()


class TestLogisticInteractions:
    """Test interaction terms."""

    def test_logistic_with_interactions(self, titanic_data):
        """Test logistic with interaction terms."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            ivar=["age:sex"],
        )
        assert "age:sex" in lr.form
        assert lr.fitted is not None


class TestLogisticSummary:
    """Test summary output."""

    def test_summary_main(self, titanic_data):
        """Test summary main output."""
        lr = logistic(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        lr.summary(plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Logistic regression (GLM)" in output
        assert "survived" in output
        assert "Yes" in output
        assert "age" in output

    def test_summary_fit(self, titanic_data):
        """Test summary fit statistics."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        lr.summary(plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Pseudo R-squared" in output
        assert "AUC" in output

    def test_summary_ci(self, titanic_data):
        """Test summary confidence intervals."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        lr.summary(ci=True, plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Confidence intervals" in output

    def test_summary_vif(self, titanic_data):
        """Test summary VIF output."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "fare"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        lr.summary(vif=True, plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Variance inflation" in output


class TestLogisticPredict:
    """Test prediction functionality."""

    def test_predict_default(self, titanic_data):
        """Test prediction with default data."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        pred = lr.predict()
        assert "prediction" in pred.columns
        assert len(pred) > 0
        # Predictions should be probabilities between 0 and 1
        assert all(pred["prediction"] >= 0)
        assert all(pred["prediction"] <= 1)

    def test_predict_new_data(self, titanic_data):
        """Test prediction with new data."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        new_data = titanic_data.head(10)
        pred = lr.predict(data=new_data)
        assert len(pred) == 10
        assert "prediction" in pred.columns

    def test_predict_with_cmd(self, titanic_data):
        """Test prediction with cmd parameter."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        pred = lr.predict(cmd={"age": [20, 40, 60]})
        assert len(pred) == 3
        assert "prediction" in pred.columns

    def test_predict_with_ci(self, titanic_data):
        """Test prediction with confidence intervals."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        pred = lr.predict(cmd={"age": [20, 40, 60]}, ci=True, conf=0.95)
        assert "prediction" in pred.columns
        assert "2.50%" in pred.columns
        assert "97.50%" in pred.columns


class TestLogisticPlot:
    """Test plotting functionality."""

    def test_plot_dist(self, titanic_data):
        """Test distribution plot."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        lr.plot(plots="dist")
        plt.savefig(f"{PLOT_DIR}/logistic_dist.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_or(self, titanic_data):
        """Test odds ratio plot."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
        )
        lr.plot(plots="or")
        plt.savefig(f"{PLOT_DIR}/logistic_or.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_pred(self, titanic_data):
        """Test prediction plot."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
        )
        lr.plot(plots="pred", incl=["age"])
        plt.savefig(f"{PLOT_DIR}/logistic_pred.png", dpi=100, bbox_inches="tight")
        plt.close("all")


class TestLogisticChisqTest:
    """Test chi-squared test for model comparison."""

    def test_chisq_test_basic(self, titanic_data):
        """Test chi-squared test for variable significance."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        lr.chisq_test(test=["age"])
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Model 1" in output
        assert "Model 2" in output
        assert "Chi-squared" in output

    def test_chisq_test_multiple_vars(self, titanic_data):
        """Test chi-squared test with multiple variables."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        lr.chisq_test(test=["age", "pclass"])
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Chi-squared" in output


class TestLogisticModelQuality:
    """Test model quality metrics."""

    def test_auc_reasonable(self, titanic_data):
        """Test that AUC is reasonable."""
        lr = logistic(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
        )
        from pyrsm.model.perf import auc
        pred_df = lr.predict()
        auc_score = auc(lr.data["survived"].to_list(), pred_df["prediction"].to_list())
        # AUC should be between 0.5 (random) and 1 (perfect)
        assert 0.5 <= auc_score <= 1
