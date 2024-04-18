from typing import Optional, Literal
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn import metrics, tree
from xgboost import XGBClassifier, XGBRegressor
from sklearn.inspection import PartialDependenceDisplay as pdp
from pyrsm.utils import ifelse, check_dataframe, setdiff
from pyrsm.model.model import (
    sim_prediction,
    convert_binary,
    evalreg,
    convert_to_list,
    conditional_get_dummies,
    reg_dashboard,
)
from pyrsm.model.perf import auc
from .visualize import pred_plot_sk, vimp_plot_sk

class xgboost:
    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: Optional[str] = None,
        lev: Optional[str] = None,
        evar: Optional[list[str]] = None,
        objective: Literal["reg", "binary:logistic"] = "reg",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1,
        colsample_bytree: float = 1,
        colsample_bylevel: float = 1,
        reg_lambda: float = 1,
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
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.kwargs = kwargs

        if self.mod_type == "classification":
            if self.lev is not None and self.rvar is not None:
                self.data = convert_binary(self.data, self.rvar, self.lev)

            self.xgb = XGBClassifier(
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **kwargs,
            )
        else:
            self.xgb = XGBRegressor(
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **kwargs,
            )
        # only use drop_first=True for a decision tree where the categorical
        # variables are only binary
        # self.data_onehot = pd.get_dummies(self.data[evar], drop_first=drop_first)
        self.data_onehot = conditional_get_dummies(self.data[evar])
        self.n_features = [len(evar), self.data_onehot.shape[1]]
        self.fitted = self.xgb.fit(self.data_onehot, self.data[self.rvar])
        self.nobs = self.data.dropna().shape[0]

    def summary(self, dec=3) -> None:
        """
        Summarize output from a XGBoost model
        """
        print("XGBoost")
        print(f"Data                 : {self.name}")
        print(f"Response variable    : {self.rvar}")
        if self.mod_type == "classification":
            print(f"Level                : {self.lev}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        print(
            f"Model type           : {ifelse(self.mod_type == 'classification', 'classification', 'regression')}"
        )
        print(f"Nr. of features      : ({self.n_features[0]}, {self.n_features[1]})")
        print(f"Nr. of observations  : {format(self.nobs, ',.0f')}")
        print(f"max_depth            : {self.max_depth}"),
        print(f"n_estimators         : {self.n_estimators}")
        print(f"subsample            : {self.subsample}")
        print(f"random_state         : {self.random_state}")
        if self.mod_type == "classification":
            print(
                f"AUC                  : {round(auc(self.data[self.rvar], self.fitted.predict_proba(self.data_onehot)[:, -1]), dec)}"
            )
        else:
            print("Model fit            :")
            print(
                evalreg(
                    pd.DataFrame().assign(
                        rvar=self.data_std[[self.rvar]],
                        prediction=self.fitted.predict(self.data_onehot),
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
        
        print("\nRaw data             :")
        print(self.data[self.evar].head().to_string(index=False))

        print("\nEstimation data      :")
        print(self.data_onehot.head().to_string(index=False))
    
    def predict(self, data=None, cmd=None, data_cmd=None) -> pd.DataFrame:
        """
        Predict probabilities or values for a XGBoost model
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

        # not dropping the first level of the categorical variables
        data_onehot = pd.get_dummies(data, drop_first=False)

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
        plots: Literal["pred", "pdp", "vimp"] = "pred",
        data=None,
        incl=None,
        excl=None,
        incl_int=[],
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
        Plots for a XGBoost model
        """
        if "pred" in plots:
            pred_plot_sk(
                self.fitted,
                data=ifelse(data is None, self.data[self.evar], data),
                rvar=self.rvar,
                incl=incl,
                excl=[],
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
            fig = pdp.from_estimator(
                self.fitted, self.data_onehot, self.data_onehot.columns, ax=ax, n_cols=2
            )

        if "vimp" in plots:
            return_vimp = vimp_plot_sk(
                self.fitted,
                self.data_onehot,
                self.data[self.rvar],
                rep=5,
                ax=ax,
                ret=True,
            )
            if ret is not None:
                return return_vimp

        if "dashboard" in plots and self.mod_type == "regression":
            model = self.fitted
            model.fittedvalues = self.predict()["prediction"]
            model.resid = self.data[self.rvar] - model.fittedvalues
            model.model = pd.DataFrame({"endog": self.data[self.rvar]})
            reg_dashboard(model, nobs=nobs)