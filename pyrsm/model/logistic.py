from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

from pyrsm.basics.correlation import correlation
from pyrsm.model.model import (
    check_binary,
    convert_to_list,
    model_fit,
    or_ci,
    or_plot,
    predict_ci,
    sig_stars,
    sim_prediction,
    to_pandas_with_categories,
)
from pyrsm.model.model import vif as calc_vif
from pyrsm.model.visualize import (
    distr_plot,
    extract_evars,
    extract_rvar,
    pred_plot_sm,
    vimp_plot_sm,
)
from pyrsm.basics import display_utils as du
from pyrsm.utils import check_dataframe, ifelse, setdiff


class logistic:
    """
    A class to perform logistic regression modeling (binary classification)

    Attributes
    ----------
    data : pl.DataFrame
        Dataset used for the analysis (stored as polars DataFrame).
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
    coef : pl.DataFrame
        The estimated model coefficients with standard errors, p-values, etc.
    weights: pl.Series or None
        Frequency weights used in the model if provided.
    weights_name: str or None
        Column name for the variable in the data containing frequency weights if provided.
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
        self.ivar = convert_to_list(ivar)
        self.form = form

        # Apply binary conversion on polars DataFrame
        if self.lev is not None and self.rvar is not None:
            self.data = check_binary(self.data, self.rvar, self.lev)

        # Handle weights
        if weights is not None and weights != "None":
            self.weights_name = weights
            self.weights = self.data.select("weights")
        else:
            self.weights_name = self.weights = None

        if self.weights is None:
            weights = None
        else:
            weights = self.weights.to_pandas()

        if self.form:
            self.fitted = smf.glm(
                formula=self.form,
                data=self.data.to_pandas(),
                freq_weights=weights,
                family=Binomial(link=Logit()),
            ).fit()
            self.evar = extract_evars(self.fitted.model, self.data.columns)
            self.rvar = extract_rvar(self.fitted.model, self.data.columns)
            if self.lev is None:
                self.lev = self.data[self.rvar][0]
        else:
            if self.evar is None or len(self.evar) == 0:
                self.form = f"{self.rvar} ~ 1"
            else:
                self.form = f"{self.rvar} ~ {' + '.join(self.evar)}"
            if self.ivar:
                self.form += f" + {' + '.join(self.ivar)}"
            self.fitted = smf.glm(
                formula=self.form,
                data=self.data.to_pandas(),
                freq_weights=weights,
                family=Binomial(link=Logit()),
            ).fit()

        self.fitted.nobs_dropped = self.data.height - self.fitted.nobs

        # Build coefficient table as polars DataFrame
        or_vals = np.exp(self.fitted.params)
        or_pct = 100 * np.where(or_vals < 1, -(1 - or_vals), or_vals - 1)
        self.coef = pl.DataFrame(
            {
                "index": self.fitted.params.index.tolist(),
                "OR": or_vals,
                "OR%": or_pct,
                "coefficient": self.fitted.params.values,
                "std.error": (self.fitted.params / self.fitted.tvalues).values,
                "z.value": self.fitted.tvalues.values,
                "p.value": self.fitted.pvalues.values,
                "  ": sig_stars(self.fitted.pvalues.values),
            }
        ).drop_nulls(subset=["coefficient"])

    def summary(self, vif=False, test=None, ci=False, dec=3, plain=False) -> None:
        """
        Summarize the logistic regression model output

        Parameters
        ----------
        vif : bool, default False
            Print the generalized variance inflation factors.
        test : list[str] or None, optional
            List of variable names to test using a Chi-Square test.
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
                print(f"\n{calc_vif(self.fitted, dec=dec).to_string()}")

        if ci:
            print("\nConfidence intervals:")
            df = or_ci(self.fitted, dec=dec).set_index("index")
            df.index.name = None
            print(f"\n{df.to_string()}")

        if test is not None and len(test) > 0:
            self.chisq_test(test=test, dec=dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        print("Logistic regression (GLM)")
        print(f"Data                 : {self.name}")
        print(f"Response variable    : {self.rvar}")
        print(f"Level                : {self.lev}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        if self.weights is not None:
            print(f"Weights used         : {self.weights_name}")
        print(f"Null hyp.: There is no effect of x on {self.rvar}")
        print(f"Alt. hyp.: There is an effect of x on {self.rvar}\n")

    def _summary_plain(self, dec: int = 3) -> None:
        """Print plain text coefficient table."""
        df = self.coef.clone()
        df = df.with_columns(
            [
                pl.col("OR").round(dec),
                pl.col("coefficient").round(dec),
                pl.col("std.error").round(dec),
                pl.col("z.value").round(dec),
                pl.when(pl.col("p.value") < 0.001)
                .then(pl.lit("< .001"))
                .otherwise(pl.col("p.value").round(dec).cast(pl.Utf8))
                .alias("p.value"),
                (pl.col("OR%").round(max(dec - 2, 0)).cast(pl.Utf8) + "%").alias("OR%"),
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
                (pl.col("OR%").round(max(dec - 2, 0)).cast(pl.Utf8) + "%").alias("OR%"),
            ]
        )

        gt = du.style_table(
            df,
            title="Coefficient Estimates",
            subtitle=f"Response: {self.rvar} (level: {self.lev})",
            number_cols=["OR", "coefficient", "std.error", "z.value"],
            dec=dec,
        )
        display(gt)

    def predict(
        self, data=None, cmd=None, data_cmd=None, ci=False, conf=0.95, dec=None
    ) -> pl.DataFrame:
        """
        Generate probability predictions using the fitted model.

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
        ret=None,
    ) -> None:
        """
        Plots for a logistic regression model
        """
        plots = convert_to_list(plots)
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
        plot_data_pd = plot_data.to_pandas()

        if "dist" in plots:
            return distr_plot(plot_data_pd)
        elif "corr" in plots:
            cr = correlation(plot_data)
            return cr.plot(nobs=nobs, figsize=figsize)
        elif "or" in plots or "coef" in plots:
            return or_plot(
                self.fitted,
                alpha=alpha,
                intercept=intercept,
                incl=incl,
                excl=excl,
                figsize=figsize,
            )
        elif "pred" in plots:
            return pred_plot_sm(
                self.fitted,
                data=plot_data_pd,
                incl=incl,
                excl=excl,
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=nnv,
                minq=minq,
                maxq=maxq,
            )
        elif "vimp" in plots or "pimp" in plots:
            (return_vimp, p) = vimp_plot_sm(
                self.fitted,
                data=plot_data_pd,
                rep=10,
                ret=ret,
            )
            if ret:
                return return_vimp
            return p

    def chisq_test(self, test=None, dec=3) -> None:
        """
        Chisq-test for competing models
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

        # Convert to pandas for statsmodels
        data_pd = self.data.to_pandas()

        sub_fitted = smf.glm(
            formula=form,
            data=data_pd,
            freq_weights=self.weights,
            family=Binomial(link=Logit()),
        ).fit()

        lrtest = -2 * (sub_fitted.llf - self.fitted.llf)
        df = self.fitted.df_model - sub_fitted.df_model
        pvalue = stats.chi2.sf(lrtest, df)

        pr2_full = 1 - self.fitted.llf / self.fitted.llnull
        pr2_sub = 1 - sub_fitted.llf / sub_fitted.llnull

        print(f"Pseudo R-squared, Model 1 vs 2: {pr2_sub:.3f} vs {pr2_full:.3f}")
        pvalue = ifelse(pvalue < 0.001, "< .001", round(pvalue, dec))
        print(f"Chi-squared: {round(lrtest, dec)} df ({df:.0f}), p.value {pvalue}")
