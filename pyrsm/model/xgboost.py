from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay as pdp

from pyrsm.model.model import (
    check_binary,
    conditional_get_dummies,
    convert_to_list,
    evalreg,
    nobs_dropped,
    reg_dashboard,
    sim_prediction,
)
from pyrsm.model.perf import auc
from pyrsm.utils import check_dataframe, ifelse, setdiff

from .visualize import pred_plot_sk, vimp_plot_sk, vimp_plot_sklearn


class xgboost:
    """
    Initialize XGBoost model

    Attributes
    ----------
    data: pandas DataFrame; dataset
    lev: String; name of the level to predict in the response variable
    rvar: String; name of the column to be used as the response variable
    evar: List of strings; contains the names of the column of data to be used as the explanatory (target) variable
    n_estimators: The number of boosting rounds
    max_depth: Maximum tree depth for base learners
    min_child_weight: Minimum sum of instance weight (hessian) needed in a child
    learning_rate: Boosting learning rate
    subsample: Subsample ratio of the training instance
    colsample_bytree: Subsample ratio of columns when constructing each tree
    random_state: Random seed used when bootstrapping data samples
    mod_type: String; type of model to be used (classification or regression)
    **kwargs : Named arguments to be passed to the XGBoost functions
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: Optional[str] = None,
        lev: Optional[str] = None,
        evar: Optional[list[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_child_weight: int = 1,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = 1234,
        mod_type: Literal["regression", "classification"] = "classification",
        **kwargs,
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        # Store as polars internally
        self.data = check_dataframe(self.data)
        self.rvar = rvar
        self.lev = lev
        self.evar = convert_to_list(evar)
        self.mod_type = mod_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.ml_model = {"model": "xgboost", "mod_type": mod_type}
        self.kwargs = kwargs
        self.nobs_all = self.data.height

        # Apply binary conversion on polars DataFrame
        if self.mod_type == "classification":
            if self.lev is not None and self.rvar is not None:
                self.data = check_binary(self.data, self.rvar, self.lev)

        # Drop nulls and get training data
        training_data = self.data.select([self.rvar] + self.evar).drop_nulls()
        self.nobs = training_data.height
        self.nobs_dropped = self.nobs_all - self.nobs

        # Convert to pandas for sklearn
        training_pd = training_data.to_pandas()
        self._rvar = training_pd[self.rvar]  # Store response for summary

        if self.mod_type == "classification":
            objective = "binary:logistic"
            self.xgb = xgb.XGBClassifier(
                objective=objective,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                **kwargs,
            )
        else:
            objective = "reg:squarederror"
            self.xgb = xgb.XGBRegressor(
                objective=objective,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                **kwargs,
            )

        # Identify categorical columns for one-hot encoding
        cat_cols = [
            c
            for c in self.evar
            if training_pd[c].dtype == "object" or training_pd[c].dtype.name == "category"
        ]

        # conditional_get_dummies returns polars DataFrame
        self.data_onehot = conditional_get_dummies(training_pd[self.evar])
        self.n_features = [len(evar), self.data_onehot.width]

        # Derive categories from dummy column names (after conditional drop_first)
        self.categories = {}
        for col in cat_cols:
            prefix = f"{col}_"
            self.categories[col] = [
                c.replace(prefix, "") for c in self.data_onehot.columns if c.startswith(prefix)
            ]

        # .to_pandas() at sklearn call site
        self.fitted = self.xgb.fit(
            self.data_onehot.to_pandas(),
            self._rvar,
            eval_set=[(self.data_onehot.to_pandas(), self._rvar)],
            verbose=False,
        )

    def summary(self, dec=3) -> None:
        """
        Summarize output from an XGBoost model

        Parameters
        ----------
        dec : int, default=3
            Number of decimal places to display in the summary.

        Examples
        --------
        >>> xgb.summary()
        """
        print("XGBoost")
        print(f"Data                 : {self.name}")
        print(f"Response variable    : {self.rvar}")
        if self.mod_type == "classification":
            print(f"Level                : {self.lev}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        print(f"Model type           : {self.mod_type}")
        print(f"Nr. of features      : ({self.n_features[0]}, {self.n_features[1]})")
        print(f"Nr. of observations  : {format(self.nobs, ',.0f')}{nobs_dropped(self)}")
        print(f"n_estimators         : {self.n_estimators}")
        print(f"max_depth            : {self.max_depth}")
        print(f"min_child_weight     : {self.min_child_weight}")
        print(f"learning_rate        : {self.learning_rate}")
        print(f"subsample            : {self.subsample}")
        print(f"colsample_bytree     : {self.colsample_bytree}")
        print(f"random_state         : {self.random_state}")

        # Get predictions (.to_pandas() at sklearn call site)
        if self.mod_type == "classification":
            pred = self.fitted.predict_proba(self.data_onehot.to_pandas())[:, -1]
            print(f"AUC                  : {round(auc(self._rvar, pred), dec)}")
        else:
            pred = self.fitted.predict(self.data_onehot.to_pandas())
            print("Model fit            :")
            print(
                evalreg(
                    pd.DataFrame().assign(
                        rvar=self._rvar,
                        prediction=pred,
                    ),
                    "rvar",
                    "prediction",
                    dec=dec,
                )
                .to_pandas()
                .T[2:]
                .rename(columns={0: " "})
                .T.to_string()
            )

        if len(self.kwargs) > 0:
            kwargs_list = [f"{k}={v}" for k, v in self.kwargs.items()]
            print(f"Extra arguments      : {', '.join(kwargs_list)}")
        print("\nEstimation data      :")
        print(self.data_onehot.head())

    def predict(self, data=None, cmd=None, data_cmd=None, dec=None) -> pl.DataFrame:
        """
        Predict probabilities or values for an XGBoost model

        Parameters
        ----------
        dec : int, optional
            Number of decimal places to round float columns in the output.
            If None (default), no rounding is applied.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the predictions and the data used to make those predictions.
        """
        if data is None:
            pred_data = self.data.select(self.evar)
        else:
            pred_data = check_dataframe(data).select(self.evar)

        if data_cmd is not None and data_cmd != "":
            pred_data = pred_data.with_columns([pl.lit(v).alias(k) for k, v in data_cmd.items()])
        elif cmd is not None and cmd != "":
            cmd = {k: ifelse(isinstance(v, str), [v], v) for k, v in cmd.items()}
            pred_data = sim_prediction(data=pred_data, vary=cmd)

        # Convert to pandas for sklearn
        pred_data_pd = pred_data.to_pandas()

        # Use categories to preserve all levels
        data_onehot = conditional_get_dummies(
            pred_data_pd, drop_nonvarying=False, categories=self.categories
        )

        # .to_pandas() at sklearn call site
        if self.mod_type == "classification":
            predictions = self.fitted.predict_proba(data_onehot.to_pandas())[:, -1]
        else:
            predictions = self.fitted.predict(data_onehot.to_pandas())

        pred = pred_data.with_columns(pl.lit(predictions).alias("prediction"))

        if dec is not None:
            pred = pred.with_columns(
                [
                    pl.col(c).round(dec)
                    for c in pred.columns
                    if pred[c].dtype in [pl.Float64, pl.Float32]
                ]
            )
        return pred

    def plot(
        self,
        plots: Literal["pred", "pdp", "vimp", "vimp_sklearn", "dashboard"] = "pred",
        data=None,
        incl=None,
        excl=None,
        incl_int=None,
        nobs: int = 1000,
        fix=True,
        hline=False,
        nnv=20,
        minq=0.025,
        maxq=0.975,
        figsize=None,
        ret=None,
    ) -> None:
        """
        Plots for an XGBoost model

        Parameters
        ----------
        plots : str or list[str], default 'pred'
            List of plot types to generate. Options include 'pred', 'pdp', 'vimp', 'vimp_sklearn', 'dashboard'.
        nobs : int, default 1000
            Number of observations to plot. Relevant for all plots that include a scatter of data points (i.e., corr, scatter, dashboard, residual).
        incl : list[str], optional
            Variables to include in the plot. Relevant for prediction plots (pred) and coefficient plots (coef).
        excl : list[str], optional
            Variables to exclude from the plot. Relevant for prediction plots (pred) and coefficient plots (coef).
        incl_int : list[str], optional
            Interaction terms to include in the plot. Relevant for prediction plots (pred).
        fix : bool, default True
            Fix the y-axis limits. Relevant for prediction plots (pred).
        hline : bool, default False
            Add a horizontal line to the plot at the mean of the response variable. Relevant for prediction plots (pred).
        nnv : int, default 20
            Number of predicted values to calculate and to plot. Relevant for prediction plots.
        minq : float, default 0.025
            Minimum quantile of the explanatory variable values to use to calculate and plot predictions.
        maxq : float, default 0.975
            Maximum quantile of the explanatory variable values to use to calculate and plot predictions.
        figsize : tuple[int, int], default None
            Figure size for the plots in inches (e.g., "(3, 6)"). Relevant for 'corr', 'scatter', 'residual', and 'coef' plots.
        ret : bool, optional
            Whether to return the variable (permutation) importance scores for a "vimp" plot.
        """
        plots = convert_to_list(plots)  # control for the case where a single string is passed
        excl = convert_to_list(excl)
        incl = ifelse(incl is None, None, convert_to_list(incl))
        incl_int = convert_to_list(incl_int)

        if data is None:
            plot_data = self.data
        else:
            plot_data = check_dataframe(data)

        # Select relevant columns
        if self.rvar in plot_data.columns:
            plot_data = plot_data.select([self.rvar] + self.evar)
        else:
            plot_data = plot_data.select(self.evar)

        # Convert to pandas for plotting (seaborn/matplotlib)
        data = plot_data.to_pandas()

        if "pred" in plots:
            return pred_plot_sk(
                self.fitted,
                data=data,
                rvar=self.rvar,
                incl=incl,
                excl=excl,
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=nnv,
                minq=minq,
                maxq=maxq,
            )

        if "pdp" in plots:
            if figsize is None:
                figsize = (8, len(self.data_onehot.columns) * 2)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("Partial Dependence Plots")
            pdp.from_estimator(
                self.fitted, self.data_onehot.to_pandas(), self.data_onehot.columns, ax=ax, n_cols=2
            )
            plt.show()
            plt.close()

        if "vimp" in plots or "pimp" in plots:
            (p, return_vimp) = vimp_plot_sk(
                self,
                rep=5,
                ret=ret,
            )

            if ret:
                return (p, return_vimp)
            else:
                return p

        if "vimp_sklearn" in plots or "pimp_sklearn" in plots:
            (p, return_vimp) = vimp_plot_sklearn(
                self.fitted,
                self.data_onehot.to_pandas(),
                self._rvar,
                rep=5,
                ret=ret,
            )

            if ret:
                return (p, return_vimp)
            else:
                return p

        if "dashboard" in plots and self.mod_type == "regression":
            model = self.fitted
            pred_df = self.predict()
            model.fittedvalues = pred_df["prediction"]
            model.resid = self._rvar.values - model.fittedvalues
            model.model = pl.DataFrame({"endog": self._rvar})
            return reg_dashboard(model, nobs=nobs)
