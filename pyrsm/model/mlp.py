from typing import Optional, Literal
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import PartialDependenceDisplay as pdp
from pyrsm.utils import ifelse, check_dataframe, setdiff
from pyrsm.model.model import (
    sim_prediction,
    check_binary,
    evalreg,
    convert_to_list,
    reg_dashboard,
    nobs_dropped,
    get_dummies,
)
from pyrsm.model.perf import auc
from pyrsm.stats import scale_df
from .visualize import pred_plot_sk, vimp_plot_sk, vimp_plot_sklearn

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
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='tanh'
        Activation function to transform the data at each node in the hidden layer.
    solver : {'lbfgs', 'sgd', 'adam'}, default='lbfgs'
        The solver for weight optimization. Note that 'adam' also uses stochastic gradient descent.
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
    >>> nn = mlp(data=df, rvar='target', evar=['feature1', 'feature2'])

    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: Optional[str] = None,
        lev: Optional[str] = None,
        evar: Optional[list[str]] = None,
        hidden_layer_sizes: tuple = (5,),
        alpha: float = 0.0001,
        activation: Literal["identity", "logistic", "tanh", "relu"] = "tanh",
        solver: Literal["lbfgs", "sgd", "adam"] = "lbfgs",
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

        # Store as polars internally
        self.data = check_dataframe(self.data)
        self.rvar = rvar
        self.lev = lev
        self.evar = convert_to_list(evar)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.activation = activation
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        self.nobs_all = self.data.height

        self.mod_type = mod_type
        self.ml_model = {"model": "mlp", "mod_type": mod_type}

        # Apply binary conversion on polars DataFrame
        if self.mod_type == "classification":
            if self.lev is not None and self.rvar is not None:
                self.data = check_binary(self.data, self.rvar, self.lev)

        # Drop nulls and get training data
        training_data = self.data.select([rvar] + self.evar).drop_nulls()
        self.nobs = training_data.height
        self.nobs_dropped = self.nobs_all - self.nobs

        # Convert to pandas for sklearn
        training_pd = training_data.to_pandas()

        if self.mod_type == "classification":
            if self.lev is None and pd.api.types.is_numeric_dtype(training_pd[self.rvar]):
                raise Exception(
                    f"Model type is set to 'Classification' but variable {self.rvar} is numeric and no value was set for 'lev' (level)."
                )

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
            training_pd[[rvar] + self.evar], sf=1, stats=True
        )

        # Identify categorical columns for one-hot encoding
        cat_cols = [
            c
            for c in self.evar
            if self.data_std[c].dtype == "object" or self.data_std[c].dtype.name == "category"
        ]

        # use drop_first=True for one-hot encoding because NN models include a bias term
        # get_dummies returns polars DataFrame
        self.data_onehot = get_dummies(self.data_std[self.evar])
        self.n_features = [len(evar), self.data_onehot.width]

        # Derive categories from dummy column names (after drop_first)
        self.categories = {}
        for col in cat_cols:
            prefix = f"{col}_"
            self.categories[col] = [
                c.replace(prefix, "") for c in self.data_onehot.columns if c.startswith(prefix)
            ]

        # Store feature order for prediction
        self.feature_names = self.data_onehot.columns

        # .to_pandas() at sklearn call site
        self.fitted = self.mlp.fit(self.data_onehot.to_pandas(), self.data_std[self.rvar])
        self.n_weights = sum(weight_matrix.size for weight_matrix in self.fitted.coefs_) + sum(
            [len(i) for i in self.fitted.intercepts_]
        )

    def summary(self, dec=3) -> None:
        """
        Summarize the output from a Multi-layer Perceptron (NN) model.

        Parameters
        ----------
        dec : int, default=3
            Number of decimal places to display in the summary.

        Examples
        --------
        >>> nn.summary()
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
        print(f"Nr. of weights       : {format(self.n_weights, ',.0f')}")
        print(f"Nr. of observations  : {format(self.nobs, ',.0f')}{nobs_dropped(self)}")
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
                f"AUC                  : {round(auc(self.data[self.rvar], self.fitted.predict_proba(self.data_onehot.to_pandas())[:, -1]), dec)}"
            )
        else:
            print("Model fit            :")
            print(
                evalreg(
                    pd.DataFrame().assign(
                        rvar=self.data_std[[self.rvar]],
                        prediction=self.fitted.predict(self.data_onehot.to_pandas()),
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

        print("\nRaw data             :")
        print(self.data.select(self.evar).head())

        print("\nEstimation data      :")
        print(self.data_onehot.head())

    def predict(
        self, data=None, cmd=None, data_cmd=None, scale=True, means=None, stds=None, dec=None
    ) -> pl.DataFrame:
        """
        Predict probabilities or values for the MLP model.

        Parameters
        ----------
        data : pd.DataFrame or pl.DataFrame, optional
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
        dec : int, optional
            Number of decimal places to round float columns in the output.
            If None (default), no rounding is applied.

        Returns
        -------
        pl.DataFrame
            DataFrame with predictions.

        Examples
        --------
        >>> predictions = model.predict(new_data)
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

        # Convert to pandas for scaling
        pred_data_pd = pred_data.to_pandas()

        if scale or (means is not None and stds is not None):
            if means is not None and stds is not None:
                data_std = scale_df(pred_data_pd, sf=1, means=means, stds=stds)
            else:
                # scaling the full dataset by the means used during estimation
                data_std = scale_df(pred_data_pd, sf=1, means=self.means, stds=self.stds)

            # Use categories to preserve all levels
            data_onehot = get_dummies(data_std, drop_nonvarying=False, categories=self.categories)
        else:
            data_onehot = get_dummies(
                pred_data_pd, drop_nonvarying=False, categories=self.categories
            )

        # Reorder columns to match training feature order
        data_onehot = data_onehot.select(self.feature_names)

        # .to_pandas() at sklearn call site
        if self.mod_type == "classification":
            predictions = self.fitted.predict_proba(data_onehot.to_pandas())[:, -1]
        else:
            predictions = (
                self.fitted.predict(data_onehot.to_pandas()) * self.stds[self.rvar]
                + self.means[self.rvar]
            )

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
        plots: Literal["pred", "pdp", "pimp", "vimp_sklearn", "dashboard"] = "pred",
        data=None,
        incl=None,
        excl=None,
        incl_int=None,
        nobs: int = 1000,
        fix=True,
        hline=True,
        nnv=30,
        minq=0.025,
        maxq=0.975,
        ret=None,
    ) -> None:
        """
        Generate plots for the Multi-layer Perceptron model.

        Parameters
        ----------
        plots : {'pred', 'pdp', 'vimp', 'dashboard'}, default='pred'
            Type of plot to generate. Options are 'pred' for prediction plot, 'pdp' for partial dependence plot, 'pimp' for variable importance plot which uses permutation importance, and 'dashboard' for a regression dashboard.
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
            Whether to fix the scale of the plots based on the maximum impact range of the included explanatory variables.
        hline : bool, default=False
            Whether to include a horizontal line at the mean response rate in the plots.
        nnv : int, default=30
            Number of values to use for the prediction plot.
        minq : float, default=0.025
            Minimum quantile for the prediction plot.
        maxq : float, default=0.975
            Maximum quantile for the prediction plot.
        ret : bool, optional
            Whether to return the variable (permutation) importance scores for a "vimp" plot.

        Examples
        --------
        >>> model.plot(plots='pdp')
        >>> model.plot(plots='pred', data=new_data)
        """

        plots = convert_to_list(plots)  # control for the case where a single string is passed
        excl = convert_to_list(excl)
        incl = ifelse(incl is None, None, convert_to_list(incl))
        incl_int = convert_to_list(incl_int)

        if "pred" in plots:
            if data is None:
                data_dct = {
                    "data": self.data.select(self.evar + [self.rvar]),
                    "means": self.means,
                    "stds": self.stds,
                }
            else:
                plot_data = check_dataframe(data)
                if self.rvar in plot_data.columns:
                    cols = self.evar + [self.rvar]
                else:
                    cols = self.evar
                data_dct = {
                    "data": plot_data.select(cols),
                    "means": self.means,
                    "stds": self.stds,
                }

            return pred_plot_sk(
                self.fitted,
                data=data_dct,
                rvar=self.rvar,
                incl=incl,
                excl=excl,
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=nnv,
                minq=minq,
                maxq=maxq,
                ret=ret,
            )

        if "pdp" in plots:
            figsize = (8, len(self.data_onehot.columns) * 2)
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("Partial Dependence Plots")
            pdp.from_estimator(
                self.fitted, self.data_onehot.to_pandas(), self.data_onehot.columns, ax=ax, n_cols=2
            )
            plt.show()
            plt.close()

        if "pimp" in plots or "vimp" in plots:
            (p, return_vimp) = vimp_plot_sk(
                self,
                rep=5,
                ret=ret,
            )

            if ret:
                return (p, return_vimp)
            return p

        if "vimp_sklearn" in plots or "pimp_sklearn" in plots:
            (p, return_vimp) = vimp_plot_sklearn(
                self.fitted,
                self.data_onehot.to_pandas(),
                self.data[self.rvar],
                rep=5,
                ret=ret,
            )

            if ret:
                return (p, return_vimp)
            return p

        if "dashboard" in plots and self.mod_type == "regression":
            model = self.fitted
            pred_df = self.predict()
            model.fittedvalues = pred_df["prediction"]
            model.resid = self._rvar.values - model.fittedvalues
            model.model = pl.DataFrame({"endog": self._rvar})
            return reg_dashboard(model, nobs=nobs)
