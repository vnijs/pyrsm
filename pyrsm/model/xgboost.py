from typing import Optional, Literal
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay as pdp
from pyrsm.utils import ifelse, check_dataframe, setdiff
from pyrsm.model.model import (
    sim_prediction,
    check_binary,
    evalreg,
    convert_to_list,
    conditional_get_dummies,
    reg_dashboard,
    nobs_dropped,
)
from pyrsm.model.perf import auc
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
        self.nobs_all = self.data.shape[0]
        self.data = self.data[[self.rvar] + self.evar].dropna()
        self.nobs = self.data.shape[0]
        self.nobs_dropped = self.nobs_all - self.nobs

        if self.mod_type == "classification":
            if self.lev is not None and self.rvar is not None:
                self.data = check_binary(self.data, self.rvar, self.lev)
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

        self.data_onehot = conditional_get_dummies(self.data[self.evar])
        self.n_features = [len(evar), self.data_onehot.shape[1]]
        self.fitted = self.xgb.fit(
            self.data_onehot, self.data[self.rvar], eval_set=[(self.data_onehot, self.data[self.rvar])], verbose=False
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

        # Get predictions
        if self.mod_type == "classification":
            pred = self.fitted.predict_proba(self.data_onehot)[:, -1]
            print(f"AUC                  : {round(auc(self.data[self.rvar], pred), dec)}")
        else:
            pred=self.fitted.predict(self.data_onehot)
            print("Model fit            :")
            print(
                evalreg(
                    pd.DataFrame().assign(
                        rvar=self.data[[self.rvar]],
                        prediction=pred,
                    ),
                    "rvar",
                    "prediction",
                    dec=dec,
                )
                .T[2:]
                .rename(columns={0: " "})
                .T.to_string()
            )

        if len(self.kwargs) > 0:
            kwargs_list = [f"{k}={v}" for k, v in self.kwargs.items()]
            print(f"Extra arguments      : {', '.join(kwargs_list)}")
        print("\nEstimation data      :")
        print(self.data_onehot.head().to_string(index=False))

    def predict(self, data=None, cmd=None, data_cmd=None) -> pd.DataFrame:
        """
        Predict probabilities or values for an XGBoost model
        """

        if data is None:
            data = self.data.loc[:, self.evar].copy()
        else:
            data = data.loc[:, self.evar].copy()

        if data_cmd is not None and data_cmd != "":
            for k, v in data_cmd.items():
                data[k] = v
        elif cmd is not None and cmd != "":
            cmd = {k: ifelse(isinstance(v, str), [v], v) for k, v in cmd.items()}
            data = sim_prediction(data=data, vary=cmd)

        # only dropping the first level for binary categorical variables
        data_onehot = conditional_get_dummies(data, drop_nonvarying=False)

        # adding back levels for categorical variables is they were removed
        if data_onehot.shape[1] != self.data_onehot.shape[1]:
            for k in setdiff(self.data_onehot.columns, data_onehot.columns):
                data_onehot[k] = False
            data_onehot = data_onehot[self.data_onehot.columns]

        if self.mod_type == "classification":
            return data.assign(prediction=self.fitted.predict_proba(data_onehot)[:, -1])
        else:
            return data.assign(prediction=self.fitted.predict(data_onehot))

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
        ax=None,
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
        ax : plt.Axes, optional
            Axes object to plot on.
        ret : bool, optional
            Whether to return the variable (permutation) importance scores for a "vimp" plot.
        """
        plots = convert_to_list(plots)  # control for the case where a single string is passed
        excl = convert_to_list(excl)
        incl = ifelse(incl is None, None, convert_to_list(incl))
        incl_int = convert_to_list(incl_int)

        if data is None:
            data = self.data
        else:
            data = check_dataframe(data)
        if self.rvar in data.columns:
            data = data[[self.rvar] + self.evar].copy()
        else:
            data = data[self.evar].copy()

        if "pred" in plots:
            pred_plot_sk(
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
            fig = pdp.from_estimator(self.fitted, self.data_onehot, self.data_onehot.columns, ax=ax, n_cols=2)

        if "vimp" in plots or "pimp" in plots:
            return_vimp = vimp_plot_sk(
                self,
                rep=5,
                ax=ax,
                ret=ret,
            )

            if ret:
                return return_vimp

        if "vimp_sklearn" in plots or "pimp_sklearn" in plots:
            return_vimp = vimp_plot_sklearn(
                self.fitted,
                self.data_onehot,
                self.data[self.rvar],
                rep=5,
                ax=ax,
                ret=ret,
            )

            if ret:
                return return_vimp

        if "dashboard" in plots and self.mod_type == "regression":
            model = self.fitted
            model.fittedvalues = self.predict()["prediction"]
            model.resid = self.data[self.rvar] - model.fittedvalues
            model.model = pd.DataFrame({"endog": self.data[self.rvar]})
            reg_dashboard(model, nobs=nobs)
