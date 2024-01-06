from typing import Optional, Literal
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from math import sqrt, log2

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay as pdp


from pyrsm.utils import ifelse, check_dataframe, setdiff
from pyrsm.model.model import sim_prediction

from .visualize import pred_plot_sk, vimp_plot_sk


class rforest:
    """
    Initialize random forest model

    Parameters
    ----------
    data: pandas DataFrame; dataset
    lev: String; name of the level in the response variable
    rvar: String; name of the column to be used as the response variable
    evar: List of strings; contains the names of the column of data to be used as the explanatory (target) variable
    n_estimators: The number of trees in the forest.


    **kwargs : Named arguments to be passed to the sklearn's Random Forest functions
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: Optional[str] = None,
        lev: Optional[str] = None,
        evar: Optional[list[str]] = None,
        n_estimators: int = 100,
        min_samples_leaf: float | int = 1,
        max_features: float | int | Literal["sqrt", "log2"] = "sqrt",
        max_samples: float = 1.0,
        oob_score: bool = True,
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
        self.evar = ifelse(isinstance(evar, str), [evar], evar)
        self.mod_type = mod_type
        self.oob_score = oob_score
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.kwargs = kwargs

        if self.mod_type == "classification":
            if self.lev is not None and self.rvar is not None:
                self.data[self.rvar] = (self.data[self.rvar] == lev).astype(int)

            self.rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                oob_score=self.oob_score,
                random_state=self.random_state,
                **kwargs,
            )
        else:
            self.rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                oob_score=self.oob_score,
                random_state=self.random_state,
                **kwargs,
            )
        # only use drop_first=True for a decision tree where the categorical
        # variables are only binary
        self.data_onehot = pd.get_dummies(self.data[evar], drop_first=False)
        self.n_features = self.data_onehot.shape[1]
        self.fitted = self.rf.fit(self.data_onehot, self.data[self.rvar])

    def summary(self, dec=3) -> None:
        """
        Summarize output from a Random Forest model
        """
        print("Random Forest")
        print(f"Data                 : {self.name}")
        print(f"Response variable    : {self.rvar}")
        if self.mod_type == "classification":
            print(f"Level                : {self.lev}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        print(f"OOB                  : {self.oob_score}")
        print(
            f"Model type           : {ifelse(self.mod_type == 'classification', 'classification', 'regression')}"
        )
        if self.max_features == "sqrt":
            nr = sqrt(self.n_features)
        elif self.max_features == "log2":
            nr = log2(self.n_features)
        elif self.max_features is None or self.max_features == "":
            nr = self.n_features
        else:
            nr = int(self.n_features)
        print(f"Number of features   : {self.n_features}"),
        print(f"max_features         : {self.max_features} ({int(nr)})"),
        print(f"n_estimators         : {self.n_estimators}"),
        print(f"min_samples_leaf     : {self.min_samples_leaf}"),
        print(f"random_state         : {self.random_state}")
        if len(self.kwargs) > 0:
            kwargs_list = [f"{k}={v}" for k, v in self.kwargs.items()]
            print(f"Extra arguments      : {', '.join(kwargs_list)}")
        print("\nEstimation data      :")
        print(self.data_onehot.head().to_string(index=False))

    def predict(self, data=None, cmd=None, data_cmd=None) -> pd.DataFrame:
        """
        Predict probabilities or values for a random forest model
        """
        if data is None:
            data = self.data.copy()
        data = data.loc[:, self.evar].copy()
        if data_cmd is not None and data_cmd != "":
            for k, v in data_cmd.items():
                data[k] = v
        elif cmd is not None and cmd != "":
            cmd = {k: ifelse(isinstance(v, str), [v], v) for k, v in cmd.items()}
            data = sim_prediction(data=data, vary=cmd)

        data_onehot = pd.get_dummies(data, drop_first=False)
        if data_onehot.shape[1] != self.data_onehot.shape[1]:
            data_onehot_missing = pd.DataFrame(
                {
                    k: [False] * data_onehot.shape[0]
                    for k in setdiff(self.data_onehot.columns, data_onehot.columns)
                }
            )
            data_onehot = pd.concat([data_onehot, data_onehot_missing], axis=1)
            data_onehot = data_onehot[self.data_onehot.columns]

        if self.mod_type == "classification":
            pred = pd.DataFrame().assign(
                prediction=self.fitted.predict_proba(data_onehot)[:, -1]
            )
        else:
            pred = pd.DataFrame().assign(prediction=self.fitted.predict(data_onehot))
        return pd.concat([data, pred], axis=1)

    def plot(
        self,
        plots: Literal["pred", "pdp", "vimp"] = "pred",
        incl=None,
        excl=None,
        incl_int=[],
        fix=True,
        hline=False,
        figsize=None,
    ) -> None:
        """
        Plots for a random forest model model
        """
        if "pred" in plots:
            pred_plot_sk(
                self.fitted,
                self.data[self.evar + [self.rvar]],
                self.rvar,
                incl=incl,
                excl=[],
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=20,
                minq=0.025,
                maxq=0.975,
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
            vimp_plot_sk(
                self.fitted,
                self.data_onehot,
                self.data[self.rvar],
                rep=5,
                ax=None,
                ret=False,
            )
