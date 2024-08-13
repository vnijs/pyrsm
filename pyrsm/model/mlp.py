from typing import Optional, Literal
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import PartialDependenceDisplay as pdp
from pyrsm.utils import ifelse, check_dataframe, setdiff
from pyrsm.model.model import (
    sim_prediction,
    convert_binary,
    evalreg,
    convert_to_list,
    reg_dashboard,
)
from pyrsm.model.perf import auc
from pyrsm.stats import scale_df
from .visualize import pred_plot_sk, vimp_plot_sk

# update docstrings using https://chatgpt.com/share/e/95faf46d-7f74-4ab7-b24b-fac0d22e6dec


class mlp:
    """
    Initialize a Multi-layer Perceptron (NN) model.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame or dict of str to pd.DataFrame or pl.DataFrame
        The dataset to be used. If a dictionary is provided, the key will be used as the dataset name.
    rvar : str, optional
        The name of the column to be used as the response variable.
    lev : str, optional
        The level in the response variable to be modeled.
    evar : list of str, optional
        The names of the columns to be used as explanatory (target) variables.
    hidden_layer_sizes : tuple, default=(5,)
        The number of neurons in the hidden layers. For example, (5,) for 5 neurons in 1 hidden layer, (5, 5) for 5 neurons in 2 hidden layers, etc.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='tanh'
        Activation function to transform the data at each node in the hidden layer.
    solver : {'lbfgs', 'sgd', 'adam'}, default='lbfgs'
        The solver for weight optimization. Note that 'adam' also uses stochastic gradient descent.
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.
    batch_size : float or str, default='auto'
        Size of minibatches for stochastic optimizers (i.e., 'sgd' or 'adam'). If 'auto', batch_size=min(200, n_samples).
    learning_rate_init : float, default=0.001
        Initial learning rate used. It controls the step-size in updating the weights. Only used when solver='sgd' or 'adam'.
    max_iter : int, default=1_000_000
        Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations.
    random_state : int, default=1234
        Seed of the pseudo-random number generator to use for shuffling the data.
    mod_type : {'regression', 'classification'}, default='classification'
        Type of model to fit, either regression or classification.
    **kwargs : dict
        Additional arguments to be passed to the sklearn's Multi-layer Perceptron functions.

    Examples
    --------
    >>> model = mlp(data=df, rvar='target', evar=['feature1', 'feature2'])

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
        max_iter: int = 1_000_000,
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
                self.data = convert_binary(self.data, self.rvar, self.lev)

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
        self.data_std, self.means, self.stds = scale_df(self.data[[rvar] + self.evar], sf=1, stats=True)
        # use drop_first=True for one-hot encoding because NN models include a bias term
        self.data_onehot = pd.get_dummies(self.data_std[self.evar], drop_first=True)
        self.n_features = [len(evar), self.data_onehot.shape[1]]

        self.fitted = self.mlp.fit(self.data_onehot, self.data_std[self.rvar])
        self.n_weights = sum(weight_matrix.size for weight_matrix in self.fitted.coefs_)
        self.nobs = self.data.dropna().shape[0]

    def summary(self, dec=3) -> None:
        """
        Summarize the output from a Multi-layer Perceptron (NN) model.

        Parameters
        ----------
        dec : int, default=3
            Number of decimal places to display in the summary.

        Examples
        --------
        >>> model.summary()
        """
        print("Multi-layer Perceptron (NN)")
        print(f"Data                 : {self.name}")
        print(f"Response variable    : {self.rvar}")
        if self.mod_type == "classification":
            print(f"Level                : {self.lev}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        print(f"Model type           : {ifelse(self.mod_type == 'classification', 'classification', 'regression')}")
        print(f"Nr. of features      : ({self.n_features[0]}, {self.n_features[1]})")
        print(f"Nr. of weights       : {format(self.n_weights, ',.0f')}")
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

    def predict(self, data=None, cmd=None, data_cmd=None, scale=True, means=None, stds=None) -> pd.DataFrame:
        """
        Predict probabilities or values for the MLP model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data to predict. If None, uses the training data.
        cmd : dict, optional
            Command dictionary to simulate predictions.
        data_cmd : dict, optional
            Command dictionary to modify the data before predictions.
        scale : bool, default=True
            Whether to scale the data before prediction.
        means : pd.Series, optional
            Means of the training data features for scaling. Will use the means used during estimation if not provided.
        stds : pd.Series, optional
            Standard deviations of the training data features for scaling. Will use the standard deviations used during estimation if not provided.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions.

        Examples
        --------
        >>> predictions = model.predict(new_data)
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

        # adding back levels for categorical variables is they were removed
        if data_onehot.shape[1] != self.data_onehot.shape[1]:
            for k in setdiff(self.data_onehot.columns, data_onehot.columns):
                data_onehot[k] = False
            data_onehot = data_onehot[self.data_onehot.columns]

        if self.mod_type == "classification":
            data["prediction"] = self.fitted.predict_proba(data_onehot)[:, -1]
        else:
            print(data_onehot.head())
            data["prediction"] = self.fitted.predict(data_onehot) * self.stds[self.rvar] + self.means[self.rvar]

        return data

    def plot(
        self,
        plots: Literal["pred", "pdp", "vimp", "dashboard"] = "pred",
        data=None,
        incl=None,
        excl=None,
        incl_int=[],
        nobs: int = 1000,
        fix=True,
        hline=False,
        nnv=40,
        minq=0.025,
        maxq=0.975,
        figsize=None,
        ax=None,
        ret=None,
    ) -> None:
        """
        Generate plots for the Multi-layer Perceptron model.

        Parameters
        ----------
        plots : {'pred', 'pdp', 'vimp', 'dashboard'}, default='pred'
            Type of plot to generate. Options are 'pred' for prediction plot, 'pdp' for partial dependence plot, 'vimp' for variable importance plot which uses permutation importance, and 'dashboard' for a regression dashboard.
        data : pd.DataFrame, optional
            Data to use for the plots. If None, uses the training data.
        incl : list of str, optional
            Variables to include in the plots.
        excl : list of str, optional
            Variables to exclude from the plots.
        incl_int : list, optional
            Interactions to include in the plots.
        nobs : int, default=1000
            Number of observations to include in the scatter plots for the dashboard plot.
        fix : bool, default=True
            Whether to fix the scale of the plots based on the maximum impact range of the include explanatory variables.
        hline : bool, default=False
            Whether to include a horizontal line at the mean response rate in the plots.
        nnv : int, default=40
            Number of values to use for the prediction plot.
        minq : float, default=0.025
            Minimum quantile for the prediction plot.
        maxq : float, default=0.975
            Maximum quantile for the prediction plot.
        figsize : tuple, optional
            Figure size for the plots.
        ax : plt.Axes, optional
            Axes object to plot on.
        ret : bool, optional
            Whether to return the the variable (permutation) importance scores.

        Examples
        --------
        >>> model.plot(plots='pdp')
        >>> model.plot(plots='vimp', data=new_data)
        """
        rvar = self.rvar
        if "pred" in plots:
            if data is None:
                data_dct = {
                    "data": self.data[self.evar + [rvar]],
                    "means": self.means,
                    "stds": self.stds,
                }
            else:
                if self.rvar in data.columns:
                    vars = self.evar + [rvar]
                    if self.mod_type == "classification":
                        data = convert_binary(data, rvar, self.lev)
                else:
                    vars = self.evar
                    rvar = None
                data_dct = {
                    "data": data[vars],
                    "means": self.means,
                    "stds": self.stds,
                }

            pred_plot_sk(
                self.fitted,
                data=data_dct,
                rvar=rvar,
                incl=incl,
                excl=ifelse(excl is None, [], excl),
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
