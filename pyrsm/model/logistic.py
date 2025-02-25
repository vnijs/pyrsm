from typing import Union, Optional
import pandas as pd
import polars as pl
import numpy as np
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
import statsmodels.formula.api as smf
from scipy import stats
from pyrsm.utils import ifelse, setdiff, check_dataframe
from pyrsm.model.visualize import distr_plot, pred_plot_sm, vimp_plot_sm, extract_evars, extract_rvar
from pyrsm.model.model import (
    sig_stars,
    model_fit,
    or_ci,
    or_plot,
    sim_prediction,
    predict_ci,
    check_binary,
    convert_to_list,
)
from pyrsm.model.model import vif as calc_vif
from pyrsm.basics.correlation import correlation


class logistic:
    """
    A class to perform logistic regression modeling (binary classification)

    Attributes
    ----------
    data : pd.DataFrame
        Dataset used for the analysis. If a Polars DataFrame is provided, it will be converted to a Pandas DataFrame.
    name : str
        Name of the dataset if provided as a dictionary.
    rvar : str
        Name of the response variable in the data.
    lev: str
        Name of the level in the response variable the model will predict.
    evar : list[str]
        List of column names of the explanatory variables.
    ivar : list[str]
        List of strings with the names of the columns included as explanatory variables (e.g., ["x1:x2", "x3:x4"])
    form : str
        Model specification formula.
    fitted : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        The fitted model.
    coef : pd.DataFrame
        The estimated model coefficients with standard errors, p-values, etc.
    weights: pd.Series or None
        Frequency weights used in the model if provided.
    weights_name: str or None
        Column name for the variable in the data containing frequency weights if provided.

    Methods
    -------
    __init__(data, rvar=None, lev=None, evar=None, ivar=None, form=None)
        Initialize the logistic class with the provided data and parameters.
    summary(main=True, fit=True, ci=False, vif=False, test=None, dec=3)
        Summarize the model output.
    plot(plot_type, nobs=1000, incl=None, excl=None, incl_int=None, fix=True, hline=False, nnv=20, minq=0.025, maxq=0.975)
        Plot for the model.
    predict(data=None, cmd=None, data_cmd=None, ci=False, conf=0.95)
        Generate predictions using the fitted regression model.
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: Optional[str] = None,
        lev: Optional[str] = None,
        evar: Optional[list[str]] = None,
        ivar: Optional[list[str]] = None,
        form: Optional[str] = None,
        weights: Optional[str] = None,
    ) -> None:
        """
        Initialize the logistic class to build a logistic regression model with the provided data and parameters.

        Parameters
        ----------
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
            Dataset used for the analysis. If a Polars DataFrame is provided, it will be converted to a Pandas DataFrame. If a dictionary is provided, the key will be used as the name of the dataset.
        rvar : str, optional
            Name of the column in the data to be used as the response variable.
        lev: str
            Name of the level in the response variable the model will predict.
        evar : list[str], optional
            List of column names in the data to use as explanatory variables.
        ivar : list[str], optional
            List of interactions to add to the model as explanatory variables (e.g., ["x1:x2", "x3:x4])
        form : str, optional
            Optional formula to use if rvar and evar are not provided.
        """
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
        self.ivar = convert_to_list(ivar)
        self.form = form
        if weights is not None and weights != "None":
            self.weights_name = weights
            self.weights = self.data[weights]
        else:
            self.weights_name = self.weights = None

        if self.lev is not None and self.rvar is not None:
            self.data = check_binary(self.data, self.rvar, self.lev)

        if self.form:
            self.fitted = smf.glm(
                formula=self.form,
                data=self.data,
                freq_weights=self.weights,
                family=Binomial(link=Logit()),
            ).fit()
            self.evar = extract_evars(self.fitted.model, self.data.columns)
            self.rvar = extract_rvar(self.fitted.model, self.data.columns)
            if self.lev is None:
                self.lev = self.data.at[0, self.rvar]
        else:
            if self.evar is None or len(self.evar) == 0:
                self.form = f"{self.rvar} ~ 1"
            else:
                self.form = f"{self.rvar} ~ {' + '.join(self.evar)}"
            if self.ivar:
                self.form += f" + {' + '.join(self.ivar)}"
            self.fitted = smf.glm(
                formula=self.form,
                data=self.data,
                freq_weights=self.weights,
                family=Binomial(link=Logit()),
            ).fit()
        self.fitted.nobs_dropped = self.data.shape[0] - self.fitted.nobs
        df = pd.DataFrame(np.exp(self.fitted.params), columns=["OR"]).dropna()
        df["OR%"] = 100 * ifelse(df["OR"] < 1, -(1 - df["OR"]), df["OR"] - 1)
        df["coefficient"] = self.fitted.params
        df["std.error"] = self.fitted.params / self.fitted.tvalues
        # statsmodels uses "t.values" as the label
        df["z.value"] = self.fitted.tvalues
        df["p.value"] = self.fitted.pvalues
        df["  "] = sig_stars(self.fitted.pvalues)
        self.coef = df.reset_index()

    def summary(
        self, main=True, fit=True, ci=False, vif=False, test=None, dec=3
    ) -> None:
        """
        Summarize the logistic regression model output

        Parameters
        ----------
        main : bool, default True
            Print the main summary. Can be useful to turn off (i.e., False) when the focus is on other metrics (e.g., VIF).
        fit : bool, default True
            Print the fit statistics. Can be useful to turn off (i.e., False) when the focus is on other metrics (e.g., VIF).
        ci : bool, default False
            Print the confidence intervals for the coefficients.
        vif : bool, default False
            Print the generalized variance inflation factors.
        test : list[str] or None, optional
            List of variable names used in the model to test using a Chi-Square test or None if no tests are performed.
        dec : int, default 3
            Number of decimal places to round to.
        """
        if main:
            print("Logistic regression (GLM)")
            print(f"Data                 : {self.name}")
            print(f"Response variable    : {self.rvar}")
            print(f"Level                : {self.lev}")
            print(f"Explanatory variables: {', '.join(self.evar)}")
            if self.weights is not None:
                print(f"Weights used         : {self.weights_name}")
            print(f"Null hyp.: There is no effect of x on {self.rvar}")
            print(f"Alt. hyp.: There is an effect of x on {self.rvar}")

            df = self.coef.copy()
            df["OR"] = df["OR"].round(dec)
            df["coefficient"] = df["coefficient"].round(dec)
            df["std.error"] = df["std.error"].round(dec)
            df["z.value"] = df["z.value"].round(dec)
            df["p.value"] = ifelse(
                df["p.value"] < 0.001, "< .001", df["p.value"].round(dec)
            )
            df["OR%"] = [f"{round(o, max(dec-2, 0))}%" for o in df["OR%"]]
            df["index"] = df["index"].str.replace("[T.", "[", regex=False)
            df = df.set_index("index")
            df.index.name = None
            print(f"\n{df.to_string()}")
            print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

        if fit:
            print(f"\n{model_fit(self.fitted, dec=dec)}")

        if ci:
            print("\nConfidence intervals:")
            df = or_ci(self.fitted, dec=dec).set_index("index")
            df.index.name = None
            print(f"\n{df.to_string()}")

        if vif:
            if self.evar is None or len(self.evar) < 2:
                print("\nVariance Inflation Factors cannot be calculated")
            else:
                print("\nVariance inflation factors:")
                print(f"\n{calc_vif(self.fitted, dec=dec).to_string()}")

        if test is not None and len(test) > 0:
            self.chisq_test(test=test, dec=dec)

    def predict(
        self, data=None, cmd=None, data_cmd=None, ci=False, conf=0.95
    ) -> pd.DataFrame:
        """
        Generate probability predictions using the fitted model.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame, optional
            Data used to generate predictions. If not provided, the estimation data will be used.
        cmd : dict[str, Union[int, list[int]]], optional
            Dictionary with the names of the columns to be used in the prediction and the values to be used.
        data_cmd : dict[str, Union[int, list[int]]], optional
            Dictionary with the names of the columns to be used in the prediction and the values to be used.
        ci : bool, default False
            Calculate confidence intervals for the predictions.
        conf : float, default 0.95
            Confidence level for the intervals.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the predictions and the data used to make those predictions.

        """
        if data is None:
            data = self.data
        else:
            data = check_dataframe(data)
        data = data.loc[:, self.evar].copy()
        if data_cmd is not None:
            for k, v in data_cmd.items():
                data[k] = v
        elif cmd is not None:
            cmd = {k: ifelse(isinstance(v, str), [v], v) for k, v in cmd.items()}
            data = sim_prediction(data=data, vary=cmd)

        if ci:
            if data_cmd is not None:
                raise ValueError(
                    "Confidence intervals not available when using the Data & Command option"
                )
            else:
                return pd.concat(
                    [data, predict_ci(self.fitted, data, conf=conf)], axis=1
                )
        else:
            return data.assign(prediction=self.fitted.predict(data))

    def plot(
        self,
        plots="or",
        data=None,
        alpha=0.05,
        nobs: int = 1000,
        intercept=False,
        incl=None,
        excl=None,
        incl_int=[],
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
        Plots for a logistic regression model

        Parameters
        ----------
        plots : str or list[str], default 'dist'
            List of plot types to generate. Options include 'dist', 'corr', 'pred', 'vimp', 'or'.
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
            Figure size for the plots in inches.
        ax : plt.Axes, optional
            Axes object to plot on.
        ret : bool, optional
            Whether to return the variable (permutation) importance scores for a "vimp" plot.

        Examples
        --------
        >>> clf_lr.plot(plots='pdp')
        >>> clf_lr.plot(plots='pred', data=new_data)
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
        if "dist" in plots:
            distr_plot(data),
        if "corr" in plots:
            cr = correlation(data)
            cr.plot(nobs=nobs, figsize=figsize)
        if "or" in plots or "coef" in plots:
            or_plot(
                self.fitted,
                alpha=alpha,
                intercept=intercept,
                incl=incl,
                excl=excl,
                figsize=figsize,
            )
        if "pred" in plots:
            pred_plot_sm(
                self.fitted,
                data=data,
                incl=incl,
                excl=excl,
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=nnv,
                minq=minq,
                maxq=maxq,
            )
        if "vimp" in plots or "pimp" in plots:
            return_vimp = vimp_plot_sm(
                self.fitted,
                data=data,
                rep=10,
                ax=ax,
                ret=ret,
            )
            if ret:
                return return_vimp

    def chisq_test(self, test=None, dec=3) -> None:
        """
        Chisq-test for competing models

        Parameters
        ----------
        test : list
            List of strings; contains the names of the columns of data to be tested
        dec : int, default 3
            Number of decimal places to round to.
        """
        if test is None:
            test = self.evar
        else:
            test = ifelse(isinstance(test, str), [test], test)

        evar = [c for c in self.evar if c not in test]
        if self.ivar is not None and len(self.ivar) > 0:
            sint = setdiff(self.ivar, test)
            test += [s for t in test for s in sint if f"I({t}" not in s and t in s]
            sint = setdiff(sint, test)
        else:
            sint = []

        form = f"{self.rvar} ~ "
        if len(evar) == 0 and len(sint) == 0:
            form += "1"
        else:
            form += f"{' + '.join(evar + sint)}"

        print(f"\nModel 1: {form}")
        print(f"Model 2: {self.form}")

        # LR test of competing models (slower than Wald but more accurate)
        sub_fitted = smf.glm(
            formula=form,
            data=self.data,
            freq_weights=self.weights,
            family=Binomial(link=Logit()),
        ).fit()

        lrtest = -2 * (sub_fitted.llf - self.fitted.llf)
        df = self.fitted.df_model - sub_fitted.df_model
        pvalue = stats.chi2.sf(lrtest, df)

        # calculate pseudo R-squared values for both models
        pr2_full = 1 - self.fitted.llf / self.fitted.llnull
        pr2_sub = 1 - sub_fitted.llf / sub_fitted.llnull

        print(f"Pseudo R-squared, Model 1 vs 2: {pr2_sub:.3f} vs {pr2_full:.3f}")
        pvalue = ifelse(pvalue < 0.001, "< .001", round(pvalue, dec))
        print(f"Chi-squared: {round(lrtest, dec)} df ({df:.0f}), p.value {pvalue}")
