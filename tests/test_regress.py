"""Tests for pyrsm.model.regress module (Linear Regression).

Salary dataset columns: salary, rank, discipline, yrs_since_phd, yrs_service, sex
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
from pyrsm.model.regress import regress

# Directory for saving plot comparisons
PLOT_DIR = "tests/plot_comparisons/model"
os.makedirs(PLOT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def salary_data():
    """Load the salary dataset and convert to pandas for statsmodels."""
    salary, _ = rsm.load_data(pkg="basics", name="salary")
    # Convert to pandas since statsmodels formula API requires pandas
    return salary.to_pandas()


@pytest.fixture(scope="module")
def salary_polars():
    """Load salary as polars for testing polars input."""
    salary, _ = rsm.load_data(pkg="basics", name="salary")
    return salary


class TestRegressBasic:
    """Basic tests for regress class initialization."""

    def test_regress_basic_fit(self, salary_data):
        """Test basic linear regression fitting."""
        reg = regress(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        assert reg.name == "salary"
        assert reg.rvar == "salary"
        assert "yrs_since_phd" in reg.evar
        assert "yrs_service" in reg.evar
        assert reg.fitted is not None
        assert hasattr(reg, "coef")

    def test_regress_polars_input(self, salary_polars):
        """Test with polars DataFrame input."""
        reg = regress(
            data=salary_polars,
            rvar="salary",
            evar=["yrs_since_phd"],
        )
        assert reg.fitted is not None

    def test_regress_pandas_input(self, salary_data):
        """Test with pandas DataFrame input."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd"],
        )
        assert reg.fitted is not None

    def test_regress_dict_input(self, salary_data):
        """Test with dict input format."""
        reg = regress(
            data={"test_salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd"],
        )
        assert reg.name == "test_salary"

    def test_regress_formula(self, salary_data):
        """Test regression with formula input."""
        reg = regress(
            data=salary_data,
            form="salary ~ yrs_since_phd + yrs_service",
        )
        assert reg.rvar == "salary"
        assert "yrs_since_phd" in reg.evar
        assert "yrs_service" in reg.evar

    def test_regress_intercept_only(self, salary_data):
        """Test intercept-only model."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=None,
        )
        assert "1" in reg.form


class TestRegressCoefficients:
    """Test coefficient output."""

    def test_coef_structure(self, salary_data):
        """Test coefficient DataFrame structure."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        assert "coefficient" in reg.coef.columns
        assert "std.error" in reg.coef.columns
        assert "t.value" in reg.coef.columns
        assert "p.value" in reg.coef.columns

    def test_coef_values_reasonable(self, salary_data):
        """Test that coefficients are reasonable."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        # Intercept should exist
        assert "Intercept" in reg.coef["index"].to_list()
        # Coefficients should be finite
        assert reg.coef["coefficient"].is_finite().all()


class TestRegressInteractions:
    """Test interaction terms."""

    def test_regress_with_interactions(self, salary_data):
        """Test regression with interaction terms."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            ivar=["yrs_since_phd:yrs_service"],
        )
        assert "yrs_since_phd:yrs_service" in reg.form
        assert reg.fitted is not None


class TestRegressSummary:
    """Test summary output."""

    def test_summary_main(self, salary_data):
        """Test summary main output."""
        reg = regress(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        reg.summary(plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Linear regression (OLS)" in output
        assert "salary" in output
        assert "yrs_since_phd" in output

    def test_summary_fit(self, salary_data):
        """Test summary fit statistics."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        reg.summary(plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "R-squared" in output

    def test_summary_ci(self, salary_data):
        """Test summary confidence intervals."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        reg.summary(ci=True, plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Confidence intervals" in output

    def test_summary_vif(self, salary_data):
        """Test summary VIF output."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        reg.summary(vif=True, plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Variance inflation" in output

    def test_summary_ssq(self, salary_data):
        """Test summary sum of squares output."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        reg.summary(ssq=True, plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Sum of squares" in output

    def test_summary_rmse(self, salary_data):
        """Test summary RMSE output."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        reg.summary(rmse=True, plain=True)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Root Mean Square Error" in output


class TestRegressPredict:
    """Test prediction functionality."""

    def test_predict_default(self, salary_data):
        """Test prediction with default data."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        pred = reg.predict()
        assert "prediction" in pred.columns
        assert len(pred) > 0

    def test_predict_new_data(self, salary_data):
        """Test prediction with new data."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        new_data = salary_data.head(10)
        pred = reg.predict(data=new_data)
        assert len(pred) == 10
        assert "prediction" in pred.columns

    def test_predict_with_cmd(self, salary_data):
        """Test prediction with cmd parameter."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        pred = reg.predict(cmd={"yrs_since_phd": [5, 10, 15]})
        assert len(pred) == 3
        assert "prediction" in pred.columns

    def test_predict_with_ci(self, salary_data):
        """Test prediction with confidence intervals."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        pred = reg.predict(cmd={"yrs_since_phd": [5, 10, 15]}, ci=True, conf=0.95)
        assert "prediction" in pred.columns
        assert "2.50%" in pred.columns
        assert "97.50%" in pred.columns


class TestRegressPlot:
    """Test plotting functionality."""

    def test_plot_dist(self, salary_data):
        """Test distribution plot."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        reg.plot(plots="dist")
        plt.savefig(f"{PLOT_DIR}/regress_dist.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_scatter(self, salary_data):
        """Test scatter plot."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        reg.plot(plots="scatter", nobs=100)
        plt.savefig(f"{PLOT_DIR}/regress_scatter.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_dashboard(self, salary_data):
        """Test dashboard plot."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        reg.plot(plots="dashboard", nobs=100)
        plt.savefig(f"{PLOT_DIR}/regress_dashboard.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_coef(self, salary_data):
        """Test coefficient plot."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        reg.plot(plots="coef")
        plt.savefig(f"{PLOT_DIR}/regress_coef.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_residual(self, salary_data):
        """Test residual plot."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        reg.plot(plots="residual", nobs=100)
        plt.savefig(f"{PLOT_DIR}/regress_residual.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_pred(self, salary_data):
        """Test prediction plot."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        reg.plot(plots="pred", incl=["yrs_since_phd"])
        plt.savefig(f"{PLOT_DIR}/regress_pred.png", dpi=100, bbox_inches="tight")
        plt.close("all")


class TestRegressFTest:
    """Test F-test for model comparison."""

    def test_f_test_basic(self, salary_data):
        """Test F-test for variable significance."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service", "rank"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        reg.f_test(test=["yrs_service"])
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Model 1" in output
        assert "Model 2" in output
        assert "F-statistic" in output

    def test_f_test_multiple_vars(self, salary_data):
        """Test F-test with multiple variables."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service", "rank"],
        )
        captured = io.StringIO()
        sys.stdout = captured
        reg.f_test(test=["yrs_service", "rank"])
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "F-statistic" in output


class TestRegressCategorical:
    """Test regression with categorical variables."""

    def test_regress_categorical_var(self, salary_data):
        """Test regression with categorical explanatory variable."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "rank"],
        )
        assert reg.fitted is not None
        # Check that dummy variables were created
        assert any("[" in str(name) for name in reg.fitted.params.index)

    def test_regress_multiple_categorical(self, salary_data):
        """Test with multiple categorical variables."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["rank", "sex"],
        )
        assert reg.fitted is not None


class TestRegressModelQuality:
    """Test model quality metrics."""

    def test_rsquared_reasonable(self, salary_data):
        """Test that R-squared is reasonable."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service", "rank"],
        )
        rsq = reg.fitted.rsquared
        assert 0 <= rsq <= 1

    def test_fvalue_positive(self, salary_data):
        """Test that F-value is positive."""
        reg = regress(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
        )
        assert reg.fitted.fvalue > 0
