"""Tests for pyrsm.model.model utility functions."""

import pytest
import numpy as np
import pandas as pd
import polars as pl
import pyrsm as rsm
from pyrsm.model.model import (
    make_train,
    convert_to_list,
    sig_stars,
    get_dummies,
    conditional_get_dummies,
    evalreg,
    sim_prediction,
    extract_evars,
    extract_rvar,
    vif,
    coef_ci,
    or_ci,
    model_fit,
)


@pytest.fixture(scope="module")
def salary_data():
    """Load the salary dataset."""
    salary, _ = rsm.load_data(pkg="basics", name="salary")
    return salary.to_pandas()


@pytest.fixture(scope="module")
def titanic_data():
    """Load the titanic dataset."""
    titanic, _ = rsm.load_data(pkg="model", name="titanic")
    return titanic.drop_nulls(subset=["age"]).to_pandas()


class TestMakeTrain:
    """Tests for make_train function."""

    def test_make_train_basic(self, salary_data):
        """Test basic train/test split."""
        train_var = make_train(salary_data, test_size=0.2)
        assert len(train_var) == len(salary_data)
        # Should have approximately 80% training
        train_pct = train_var.mean()
        assert 0.7 < train_pct < 0.9

    def test_make_train_stratified(self, salary_data):
        """Test stratified train/test split."""
        train_var = make_train(salary_data, strat_var="sex", test_size=0.2)
        assert len(train_var) == len(salary_data)

    def test_make_train_polars_input(self):
        """Test with polars input."""
        salary, _ = rsm.load_data(pkg="basics", name="salary")
        train_var = make_train(salary, test_size=0.2)
        assert len(train_var) == len(salary)

    def test_make_train_reproducible(self, salary_data):
        """Test that random_state makes it reproducible."""
        train1 = make_train(salary_data, test_size=0.2, random_state=42)
        train2 = make_train(salary_data, test_size=0.2, random_state=42)
        # Stratified splits should be identical
        train1_strat = make_train(salary_data, strat_var="sex", test_size=0.2, random_state=42)
        train2_strat = make_train(salary_data, strat_var="sex", test_size=0.2, random_state=42)
        assert np.array_equal(train1_strat, train2_strat)


class TestConvertToList:
    """Tests for convert_to_list function."""

    def test_convert_none(self):
        """Test converting None."""
        assert convert_to_list(None) == []

    def test_convert_string(self):
        """Test converting single string."""
        assert convert_to_list("test") == ["test"]

    def test_convert_list(self):
        """Test converting list (should return unchanged)."""
        assert convert_to_list(["a", "b"]) == ["a", "b"]

    def test_convert_tuple(self):
        """Test converting tuple."""
        result = convert_to_list(("a", "b"))
        assert result == ["a", "b"]


class TestSigStars:
    """Tests for sig_stars function."""

    def test_sig_stars_highly_significant(self):
        """Test very small p-values."""
        result = sig_stars([0.0001, 0.0005])
        assert result.to_list() == ["***", "***"]

    def test_sig_stars_significant(self):
        """Test significant p-values."""
        result = sig_stars([0.005, 0.03, 0.08])
        assert result.to_list() == ["**", "*", "."]

    def test_sig_stars_not_significant(self):
        """Test non-significant p-values."""
        result = sig_stars([0.5, 0.9])
        assert result.to_list() == [" ", " "]

    def test_sig_stars_nan(self):
        """Test handling NaN values."""
        result = sig_stars([np.nan, 0.005])
        assert result[0] == " "  # NaN treated as p=1
        assert result[1] == "**"


class TestGetDummies:
    """Tests for get_dummies function."""

    def test_get_dummies_basic(self, salary_data):
        """Test basic dummy variable creation."""
        df = salary_data[["salary", "rank", "sex"]].copy()
        result = get_dummies(df)
        # Numeric column should remain
        assert "salary" in result.columns
        # Categorical columns should be converted
        assert any("rank_" in col for col in result.columns)
        assert any("sex_" in col for col in result.columns)

    def test_get_dummies_drop_first(self, salary_data):
        """Test drop_first parameter."""
        df = salary_data[["salary", "sex"]].copy()
        result = get_dummies(df, drop_first=True)
        # Should have n-1 dummy columns for each categorical
        sex_dummies = [col for col in result.columns if "sex_" in col]
        assert len(sex_dummies) == 1  # Male or Female, not both


class TestEvalreg:
    """Tests for evalreg function."""

    def test_evalreg_basic(self, salary_data):
        """Test regression evaluation metrics."""
        df = salary_data.copy()
        # Create a simple prediction
        df["pred"] = df["salary"].mean()
        result = evalreg(df, rvar="salary", pred="pred")
        assert "r2" in result.columns
        assert "mse" in result.columns
        assert "mae" in result.columns

    def test_evalreg_dict_input(self, salary_data):
        """Test with dict input for train/test splits."""
        df = salary_data.copy()
        df["pred"] = df["salary"].mean()
        train = df.head(300)
        test = df.tail(97)
        dct = {"train": train, "test": test}
        result = evalreg(dct, rvar="salary", pred="pred")
        assert len(result) == 2
        assert "train" in result["Type"].to_list()
        assert "test" in result["Type"].to_list()


class TestSimPrediction:
    """Tests for sim_prediction function."""

    def test_sim_prediction_default(self, salary_data):
        """Test default simulation (mode/mean values)."""
        df = salary_data[["yrs_since_phd", "yrs_service", "rank"]].copy()
        result = sim_prediction(df)
        assert len(result) == 1
        # Should have mean for numeric columns
        assert "yrs_since_phd" in result.columns

    def test_sim_prediction_vary_numeric(self, salary_data):
        """Test varying a numeric variable."""
        df = salary_data[["yrs_since_phd", "yrs_service"]].copy()
        result = sim_prediction(df, vary=["yrs_since_phd"], nnv=5)
        assert len(result) == 5

    def test_sim_prediction_vary_dict(self, salary_data):
        """Test with explicit dict values."""
        df = salary_data[["yrs_since_phd", "yrs_service"]].copy()
        result = sim_prediction(df, vary={"yrs_since_phd": [5, 10, 15, 20]})
        assert len(result) == 4


class TestVIF:
    """Tests for VIF calculation."""

    def test_vif_basic(self, salary_data):
        """Test VIF calculation."""
        import statsmodels.formula.api as smf
        fitted = smf.ols("salary ~ yrs_since_phd + yrs_service", data=salary_data).fit()
        result = vif(fitted)
        assert "vif" in result.columns
        assert "Rsq" in result.columns
        # VIF should be >= 1
        assert all(result["vif"] >= 1)

    def test_vif_with_categorical(self, salary_data):
        """Test VIF with categorical variables."""
        import statsmodels.formula.api as smf
        fitted = smf.ols("salary ~ yrs_since_phd + rank", data=salary_data).fit()
        result = vif(fitted)
        assert len(result) >= 2


class TestCoefCI:
    """Tests for coefficient confidence intervals."""

    def test_coef_ci_basic(self, salary_data):
        """Test coefficient CI calculation."""
        import statsmodels.formula.api as smf
        fitted = smf.ols("salary ~ yrs_since_phd + yrs_service", data=salary_data).fit()
        result = coef_ci(fitted)
        assert "coefficient" in result.columns
        assert "2.5%" in result.columns
        assert "97.5%" in result.columns

    def test_coef_ci_no_intercept(self, salary_data):
        """Test excluding intercept."""
        import statsmodels.formula.api as smf
        fitted = smf.ols("salary ~ yrs_since_phd + yrs_service", data=salary_data).fit()
        result = coef_ci(fitted, intercept=False)
        assert "Intercept" not in result.index


class TestModelFit:
    """Tests for model_fit function."""

    def test_model_fit_regression(self, salary_data):
        """Test model fit stats for linear regression."""
        import statsmodels.formula.api as smf
        fitted = smf.ols("salary ~ yrs_since_phd + yrs_service", data=salary_data).fit()
        result = model_fit(fitted, prn=False)
        assert "rsq" in result.columns
        assert "rsq_adj" in result.columns
        assert "fvalue" in result.columns

    def test_model_fit_logistic(self, titanic_data):
        """Test model fit stats for logistic regression."""
        import statsmodels.formula.api as smf
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.families.links import Logit
        # Convert survived to binary
        titanic_data = titanic_data.copy()
        titanic_data["survived_bin"] = (titanic_data["survived"] == "Yes").astype(int)
        fitted = smf.glm(
            "survived_bin ~ age + sex",
            data=titanic_data,
            family=Binomial(link=Logit())
        ).fit()
        result = model_fit(fitted, prn=False)
        assert "pseudo_rsq_mcf" in result.columns
        assert "AUC" in result.columns
        assert "AIC" in result.columns


class TestOrCI:
    """Tests for odds ratio confidence intervals."""

    def test_or_ci_basic(self, titanic_data):
        """Test OR CI calculation."""
        import statsmodels.formula.api as smf
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.families.links import Logit
        titanic_data = titanic_data.copy()
        titanic_data["survived_bin"] = (titanic_data["survived"] == "Yes").astype(int)
        fitted = smf.glm(
            "survived_bin ~ age + sex",
            data=titanic_data,
            family=Binomial(link=Logit())
        ).fit()
        result = or_ci(fitted)
        assert "OR" in result.columns
        assert "OR%" in result.columns
        assert "2.5%" in result.columns
        assert "97.5%" in result.columns
