from typing import Optional, Literal
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import PartialDependenceDisplay as pdp
from pyrsm.utils import ifelse, check_dataframe, setdiff
from pyrsm.model.model import sim_prediction, convert_binary, evalreg, convert_to_list
from pyrsm.model.perf import auc
from pyrsm.stats import scale_df
from .visualize import pred_plot_sk, vimp_plot_sk


class mlp:
    """
    Initialize Multi-layer Perceptron (NN) model

    Parameters
    ----------
    data: pandas DataFrame; dataset
    lev: String; name of the level in the response variable
    rvar: String; name of the column to be used as the response variable
    evar: List of strings; contains the names of the column of data to be used as the explanatory (target) variable
    hidden_layer_sizes: The number of neurons in the hidden layer and the number of hidden layers (e..g, (5,) for 5 neurons in 1 hidden layer, (5, 5) for 5 neurons in 2 hidden layers, etc.)
    activation: Activation function apply to transform for the nodes in the hidden layer
    solver: The solver used for weight optimization
    alpha: L2 penalty (regularization term) parameter

    **kwargs : Named arguments to be passed to the sklearn's Multi-layer Perceptron functions
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: Optional[str] = None,
        lev: Optional[str] = None,
        evar: Optional[list[str]] = None,
        hidden_layer_sizes: tuple = (5,),
        activation: Literal["identity", "logistic", "tanh", "relu"] = "tanh",
        solver: Literal["lbfgs", "sgd", "adam"] = "lbfgs",
        alpha: float = 0.0001,
        batch_size: float | str = "auto",
        learning_rate_init: float = 0.001,
        max_iter: int = 10_000,
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
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs

        if self.mod_type == "classification":
            if self.lev is not None and self.rvar is not None:
                self.data[self.rvar] = convert_binary(self.data[self.rvar], self.lev)

            self.mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                **kwargs,
            )
        else:
            self.mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                **kwargs,
            )
        self.data_std, self.means, self.stds = scale_df(
            self.data[[rvar] + self.evar], sf=1, stats=True
        )
        # use drop_first=True
        self.data_onehot = pd.get_dummies(self.data_std[self.evar], drop_first=True)
        self.n_features = [len(evar), self.data_onehot.shape[1]]
        self.fitted = self.mlp.fit(self.data_onehot, self.data_std[self.rvar])
        self.nobs = self.data.dropna().shape[0]

    def summary(self, dec=3) -> None:
        """
        Summarize output from a Multi-layer Perceptron (NN) model
        """
        print("Multi-layer Perceptron (NN)")
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
        print(f"Hidden_layer_sizes   : {self.hidden_layer_sizes}")
        print(f"Activation function  : {self.activation}")
        print(f"Solver               : {self.solver}")
        print(f"Alpha                : {self.alpha}")
        print(f"Batch size           : {self.batch_size}")
        print(f"Learning rate        : {self.learning_rate_init}")
        print(f"Maximum iterations   : {self.max_iter}")
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

    def predict(
        self, data=None, cmd=None, data_cmd=None, scale=True, means=None, stds=None
    ) -> pd.DataFrame:
        """
        Predict probabilities or values for an MLP
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

        if scale or (means is not None and stds is not None):
            if means is not None and stds is not None:
                data_std = scale_df(data, sf=1, means=means, stds=stds)
            else:
                # scaling the full dataset by the means used during estimation
                data_std = scale_df(data, sf=1, means=self.means, stds=self.stds)

            data_onehot = pd.get_dummies(data_std, drop_first=False)
        else:
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
            data["prediction"] = self.fitted.predict_proba(data_onehot)[:, -1]
        else:
            data["prediction"] = (
                self.fitted.predict(data_onehot) * self.stds[self.rvar]
                + self.means[self.rvar]
            )

        return data

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
        Plots for a Multi-layer Perceptron model (NN)
        """
        if "pred" in plots:
            pred_plot_sk(
                self.fitted,
                # self.data[[self.rvar] + self.evar],
                self.data_std[[self.rvar] + self.evar],
                self.rvar,
                incl=incl,
                excl=ifelse(excl is None, [], excl),
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=40,
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
