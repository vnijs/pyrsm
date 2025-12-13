import re
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

from pyrsm.basics import display_utils as du
from pyrsm.basics.correlation import correlation
from pyrsm.model.model import (
    coef_ci,
    coef_plot,
    convert_to_list,
    extract_evars,
    extract_rvar,
    model_fit,
    predict_ci,
    reg_dashboard,
    residual_plot,
    scatter_plot,
    sig_stars,
    sim_prediction,
    to_pandas_with_categories,
)
from pyrsm.model.model import vif as calc_vif
from pyrsm.model.visualize import distr_plot, pred_plot_sm, vimp_plot_sm
from pyrsm.utils import check_dataframe, format_nr, ifelse, setdiff


class regress:
    """
    A class to perform linear regression modeling

    Attributes
    ----------
    data : pd.DataFrame
        The data used in the analysis should be a Pandas or Polars DataFrame. Can also pass in a dictionary with the name of the DataFrame as the key and the DataFrame as the value.
    name : str
        The name of the dataset if provided as a dictionary.
    rvar : str
        The response variable.
    evar : list[str]
        List of column names of the explanatory variables.
    ivar : list[str]
        List of strings with the names of the columns included as explanatory variables (e.g., ["x1:x2", "x3:x4"])
    formula : str
        Model specification formula.
    fitted : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model.
    coef : pd.DataFrame
        The estimated model coefficients with standard errors, p-values, etc.

    Methods
    -------
    __init__(data, rvar=None, evar=None, ivar=None, form=None)
        Initialize the regress class with the provided data and parameters.
    summary(main=True, fit=True, ci=False, ssq=False, rmse=False, vif=False, test=None, dec=3)
        Summarize the model output.
    plot(plot_type, nobs=1000, incl=None, excl=None, incl_int=None, fix=True, hline=False, nnv=20, minq=0.025, maxq=0.975)
        Plots for the model.
    predict(data=None, cmd=None, data_cmd=None, ci=False, conf=0.95)
        Generate predictions using the fitted model.
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: str | None = None,
        evar: list[str] | None = None,
        ivar: list[str] | None = None,
        formula: str | None = None,
    ) -> None:
        """
        Initialize the regress class to build a linear regression model with the provided data and parameters.

        Parameters
        ----------
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
            Dataset used for the analysis. If a Polars DataFrame is provided, it will be converted to a Pandas DataFrame. If a dictionary is provided, the key will be used as the name of the dataset.
        rvar : str, optional
            Name of the column in the data to be use as the response variable.
        evar : list[str], optional
            List of column names in the data to use as explanatory variables.
        ivar : list[str], optional
            List of interactions to add to the model as explanatory variables (e.g., ["x1:x2", "x3:x4])
        formula : str, optional
            Optional formula to use if rvar and evar are not provided.
        """
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        # Store as polars internally
        self.data = check_dataframe(self.data)
        self.rvar = rvar
        self.evar = convert_to_list(evar)
        self.ivar = convert_to_list(ivar)
        self.formula = formula

        # Convert to pandas only for statsmodels fitting
        data_pd = self.data.to_pandas()

        if self.formula:
            self.fitted = smf.ols(formula=self.formula, data=data_pd).fit()
            self.evar = extract_evars(self.fitted.model, self.data.columns)
            self.rvar = extract_rvar(self.fitted.model, self.data.columns)
        else:
            if self.evar is None or len(self.evar) == 0:
                self.formula = f"{self.rvar} ~ 1"
            else:
                self.formula = f"{self.rvar} ~ {' + '.join(self.evar)}"
            if self.ivar:
                self.formula += f" + {' + '.join(self.ivar)}"
            self.fitted = smf.ols(self.formula, data=data_pd).fit()

        self.fitted.nobs_dropped = self.data.height - self.fitted.nobs

        # Build coefficient table as polars DataFrame
        self.coef = pl.DataFrame(
            {
                "index": self.fitted.params.index.tolist(),
                "coefficient": self.fitted.params.values,
                "std.error": (self.fitted.params / self.fitted.tvalues).values,
                "t.value": self.fitted.tvalues.values,
                "p.value": self.fitted.pvalues.values,
                "  ": sig_stars(self.fitted.pvalues.values),
            }
        ).drop_nulls(subset=["coefficient"])

    def summary(
        self,
        vif=False,
        ssq=False,
        rmse=False,
        test: str | None = None,
        ci=False,
        dec: int = 3,
        plain: bool = True,
    ) -> None:
        """
        Summarize the linear regression model output

        Parameters
        ----------
        vif : bool, default False
            Print the generalized variance inflation factors.
        ssq : bool, default False
            Print the sum of squares.
        rmse : bool, default False
            Print the root mean square error.
        test : list[str] or None, optional
            List of variable names used in the model to test using a Chi-Square test or None if no tests are performed.
        ci : bool, default False
            Print the confidence intervals for the coefficients.
        dec : int, default 3
            Number of decimal places to round to.
        plain : bool, default False
            Force plain text output (useful for non-notebook environments like Shiny).
        """

        self._summary_header()
        if not plain and du.is_notebook():
            self._summary_styled(dec)
        else:
            self._summary_plain(dec)
        du.print_sig_codes()

        print(f"\n{model_fit(self.fitted, dec=dec)}")

        if vif:
            if self.evar is None or len(self.evar) < 2:
                print("\nVariance Inflation Factors cannot be calculated")
            else:
                print("\nVariance inflation factors:")
                print(f"\n{calc_vif(self.fitted).to_string()}")

        if ssq:
            print("\nSum of squares:")
            index = ["Regression", "Error", "Total"]
            sum_of_squares = [
                self.fitted.ess,
                self.fitted.ssr,
                self.fitted.centered_tss,
            ]
            sum_of_squares = pd.DataFrame(index=index).assign(
                df=format_nr(
                    [
                        self.fitted.df_model,
                        self.fitted.df_resid,
                        self.fitted.df_model + self.fitted.df_resid,
                    ],
                    dec=0,
                ),
                SS=format_nr(sum_of_squares, dec=0),
            )
            print(f"\n{sum_of_squares.to_string()}")

        if rmse:
            print("\nRoot Mean Square Error (RMSE):")
            rmse_val = (self.fitted.ssr / self.fitted.nobs) ** 0.5
            print(round(rmse_val, dec))

        if test is not None and len(test) > 0:
            self.f_test(test=test, dec=dec)

        if ci:
            print("\nConfidence intervals:")
            df = coef_ci(self.fitted, dec=dec)
            print(f"\n{df.to_string()}")

    def _summary_header(self) -> None:
        """Print the summary header."""
        print("Linear regression (OLS)")
        print("Data                 :", self.name)
        print("Response variable    :", self.rvar)
        print("Explanatory variables:", ", ".join(self.evar))
        print(f"Null hyp.: the effect of x on {self.rvar} is zero")
        print(f"Alt. hyp.: the effect of x on {self.rvar} is not zero\n")

    def _summary_plain(self, dec: int = 3) -> None:
        """Print plain text coefficient table."""
        df = self.coef.clone()
        df = df.with_columns(
            [
                pl.col("coefficient").round(dec),
                pl.col("std.error").round(dec),
                pl.col("t.value").round(dec),
                pl.when(pl.col("p.value") < 0.001)
                .then(pl.lit("< .001"))
                .otherwise(pl.col("p.value").round(dec).cast(pl.Utf8))
                .alias("p.value"),
                pl.col("index").str.replace("[T.", "[", literal=True),
            ]
        )
        du.print_plain_tables(df)

    def _summary_styled(self, dec: int = 3) -> None:
        """Display styled coefficient table in notebooks."""
        from IPython.display import display

        df = self.coef.clone()
        df = df.with_columns(
            [
                pl.col("index").str.replace("[T.", "[", literal=True),
                pl.col("p.value").map_elements(
                    lambda p: du.format_pval(p, dec), return_dtype=pl.Utf8
                ),
            ]
        )

        gt = du.style_table(
            df,
            title="Coefficient Estimates",
            subtitle=f"Response: {self.rvar}",
            number_cols=["coefficient", "std.error", "t.value"],
            dec=dec,
        )
        display(gt)

    def predict(
        self, data=None, cmd=None, data_cmd=None, ci=False, conf=0.95, dec=None
    ) -> pl.DataFrame:
        """
        Generate predictions using the fitted model.

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
            data = check_dataframe(data)
            pred_data = data.select(self.evar)

        if data_cmd is not None:
            # Apply data_cmd values to pred_data
            pred_data = pred_data.with_columns([pl.lit(v).alias(k) for k, v in data_cmd.items()])
        elif cmd is not None:
            cmd = {k: ifelse(isinstance(v, str), [v], v) for k, v in cmd.items()}
            pred_data = sim_prediction(data=pred_data, vary=cmd)

        # Convert to pandas for statsmodels (patsy requires consistent categorical levels)
        pred_data_pd = to_pandas_with_categories(pred_data, self.data.select(self.evar))

        if ci:
            if data_cmd is not None:
                raise ValueError(
                    "Confidence intervals not available when using the Data & Command option"
                )
            else:
                # predict_ci returns pandas, convert result to polars
                ci_result = predict_ci(self.fitted, pred_data_pd, conf=conf)
                pred = pl.concat([pred_data, pl.from_pandas(ci_result)], how="horizontal")
        else:
            # Get predictions from statsmodels (needs pandas), add to polars DataFrame
            predictions = self.fitted.predict(pred_data_pd)
            pred = pred_data.with_columns(pl.lit(predictions.values).alias("prediction"))

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
        plots="dist",
        data=None,
        alpha=0.05,
        nobs: int = 1000,
        intercept=False,
        incl=None,
        excl=None,
        incl_int=None,
        fix=True,
        hline=False,
        nnv=20,
        minq=0.025,
        maxq=0.975,
        figsize=None,
        ret=None,
    ) -> None:
        """
        Plots for a linear regression model

        Parameters
        ----------
        plots : str or list[str], default 'dist'
            List of plot types to generate. Options include 'dist', 'corr', 'scatter', 'dashboard', 'residual', 'pred', 'vimp', 'coef'.
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

        if "dist" in plots:
            return distr_plot(data)
        if "corr" in plots:
            cr = correlation(data)
            return cr.plot(nobs=nobs, figsize=figsize)
        if "scatter" in plots:
            return scatter_plot(self.fitted, data, nobs=nobs, figsize=figsize)
        if "dashboard" in plots:
            return reg_dashboard(self.fitted, nobs=nobs)
        if "residual" in plots:
            return residual_plot(self.fitted, data, nobs=nobs, figsize=figsize)
        if "pred" in plots:
            return pred_plot_sm(
                self.fitted,
                data=data[self.evar],
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
            (return_vimp, p) = vimp_plot_sm(
                self.fitted,
                data=data,
                rep=10,
                ret=ret,
            )
            if ret:
                return return_vimp
            return p
        if "coef" in plots:
            return coef_plot(
                self.fitted,
                alpha=alpha,
                intercept=intercept,
                incl=incl,
                excl=excl,
                figsize=figsize,
            )

    def f_test(self, test: Optional[str] = None, dec: int = 3) -> None:
        """
        F-test for competing models

        Parameters
        ----------
        test : list[str] or None, optional
            List of variable names used in the model to test using an F-test or None if all variables are to be tested.
        dec : int, default 3
            Number of decimal places to round to.
        """
        evar = setdiff(self.evar, test)
        if self.ivar is not None and len(self.ivar) > 0:
            sint = setdiff(self.ivar, test)
            test += [s for t in test for s in sint if f"I({t}" not in s and t in s]
            sint = setdiff(sint, test)
        else:
            sint = []

        formula = f"{self.rvar} ~ "
        if len(evar) == 0 and len(sint) == 0:
            formula += "1"
        else:
            formula += f"{' + '.join(evar + sint)}"
        pattern = r"(\[T\.[^\]]*\])\:"

        # ensure constraints are unique
        hypothesis = list(
            set(
                [
                    f"({c} = 0)"
                    for c in self.fitted.model.exog_names
                    for v in test
                    if f"{v}:" in c
                    or f":{v}" in c
                    or f"{v}[T." in c
                    or v == c
                    or v == re.sub(pattern, ":", c)
                ]
            )
        )

        print(f"\nModel 1: {formula}")
        print(f"Model 2: {self.formula}")
        out = self.fitted.f_test(hypothesis)

        r2_sub = self.fitted.rsquared - (
            len(hypothesis) * out.fvalue * (1 - self.fitted.rsquared)
        ) / (self.fitted.nobs - self.fitted.df_model - 1)

        pvalue = ifelse(out.pvalue < 0.001, "< .001", round(out.pvalue, dec))
        print(f"R-squared, Model 1 vs 2: {r2_sub:.3f} vs {self.fitted.rsquared:.3f}")
        print(
            f"F-statistic: {round(out.fvalue, dec)} df ({out.df_num:.0f}, {out.df_denom:.0f}), p.value {pvalue}"
        )
