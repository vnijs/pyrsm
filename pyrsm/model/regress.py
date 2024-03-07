import pandas as pd
import polars as pl
import re
import statsmodels.formula.api as smf
from typing import Optional, Union
from pyrsm.utils import ifelse, format_nr, setdiff, check_dataframe
from pyrsm.model.visualize import pred_plot_sm, vimp_plot_sm
from pyrsm.model.model import (
    sig_stars,
    model_fit,
    extract_evars,
    extract_rvar,
    scatter_plot,
    reg_dashboard,
    residual_plot,
    coef_plot,
    coef_ci,
    predict_ci,
    sim_prediction,
    convert_to_list,
)
from pyrsm.model.model import vif as calc_vif
from pyrsm.model.visualize import distr_plot
from pyrsm.basics.correlation import correlation


class regress:
    """
    Estimate linear regression model

    Parameters
    ----------
    data: pandas DataFrame; dataset
    evar: List of strings; contains the names of the columns of data to be used as explanatory variables
    rvar: String; name of the column to be used as the response variable
    ivar: List of strings; contains the names of columns to interact and add as explanatory variables (e.g., ["x1:x2", "x3:x4])
    form: String; formula for the regression equation to use if evar and rvar are not provided
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: Optional[str] = None,
        evar: Optional[list[str]] = None,
        ivar: Optional[list[str]] = None,
        form: Optional[str] = None,
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name].copy()  # needed with pandas
        else:
            self.data = data  # needed with pandas
            self.name = "Not provided"
        self.data = check_dataframe(self.data)
        self.rvar = rvar
        self.evar = convert_to_list(evar)
        self.ivar = convert_to_list(ivar)
        self.form = form

        if self.form:
            self.fitted = smf.ols(formula=self.form, data=self.data).fit()
            self.evar = extract_evars(self.fitted.model, self.data.columns)
            self.rvar = extract_rvar(self.fitted.model, self.data.columns)
        else:
            if self.evar is None or len(self.evar) == 0:
                self.form = f"{self.rvar} ~ 1"
            else:
                self.form = f"{self.rvar} ~ {' + '.join(self.evar)}"
            if self.ivar:
                self.form += f" + {' + '.join(self.ivar)}"
            self.fitted = smf.ols(self.form, data=self.data).fit()

        df = pd.DataFrame(self.fitted.params, columns=["coefficient"]).dropna()
        df["std.error"] = self.fitted.params / self.fitted.tvalues
        df["t.value"] = self.fitted.tvalues
        df["p.value"] = self.fitted.pvalues
        df["  "] = sig_stars(self.fitted.pvalues)
        self.coef = df.reset_index()

    def summary(
        self,
        main=True,
        fit=True,
        ci=False,
        ssq=False,
        rmse=False,
        vif=False,
        test=None,
        dec=3,
    ) -> None:
        """
        Summarize output from a linear regression model

        parameters
        ----------
        ssq: Boolean; if True, include sum of squares
        vif: Boolean; if True, include variance inflation factors
        """
        if main:
            print("Linear regression (OLS)")
            print("Data                 :", self.name)
            print("Response variable    :", self.rvar)
            print("Explanatory variables:", ", ".join(self.evar))
            print(f"Null hyp.: the effect of x on {self.rvar} is zero")
            print(f"Alt. hyp.: the effect of x on {self.rvar} is not zero")

            df = self.coef.copy()
            df["coefficient"] = df["coefficient"].round(dec)
            df["std.error"] = df["std.error"].round(dec)
            df["t.value"] = df["t.value"].round(dec)
            df["p.value"] = ifelse(
                df["p.value"] < 0.001, "< .001", df["p.value"].round(dec)
            )
            df["index"] = df["index"].str.replace("[T.", "[", regex=False)
            df = df.set_index("index")
            df.index.name = None
            print(f"\n{df.to_string()}")
            print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

        if fit:
            print(f"\n{model_fit(self.fitted, dec=dec)}")

        if ci:
            print("\nConfidence intervals:")
            df = coef_ci(self.fitted, dec=dec)
            print(f"\n{df.to_string()}")

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
            rmse = (self.fitted.ssr / self.fitted.nobs) ** 0.5
            print(round(rmse, dec))

        if vif:
            if self.evar is None or len(self.evar) < 2:
                print("\nVariance Inflation Factors cannot be calculated")
            else:
                print("\nVariance inflation factors:")
                print(f"\n{calc_vif(self.fitted).to_string()}")

        if test is not None and len(test) > 0:
            self.f_test(test=test, dec=dec)

    def predict(
        self, data=None, cmd=None, data_cmd=None, ci=False, conf=0.95
    ) -> pd.DataFrame:
        """
        Predict values for a linear regression model
        """
        if data is None:
            data = self.data
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
        plots="dist",
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
        Plots for a linear regression model
        """
        data = self.data[[self.rvar] + self.evar].copy()
        if "dist" in plots:
            distr_plot(data)
        if "corr" in plots:
            cr = correlation(data)
            cr.plot(nobs=nobs, figsize=figsize)
        if "scatter" in plots:
            scatter_plot(self.fitted, data, nobs=nobs, figsize=figsize)
        if "dashboard" in plots:
            reg_dashboard(self.fitted, nobs=nobs)
        if "residual" in plots:
            residual_plot(self.fitted, data, nobs=nobs, figsize=figsize)
        if "pred" in plots:
            pred_plot_sm(
                self.fitted,
                data=ifelse(data is None, self.data[self.evar], data),
                incl=incl,
                excl=ifelse(excl is None, [], excl),
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=nnv,
                minq=minq,
                maxq=maxq,
            )
        if "vimp" in plots:
            return_vimp = vimp_plot_sm(
                self.fitted,
                data=ifelse(data is None, self.data, data),
                rep=10,
                ax=ax,
                ret=True,
            )
            if ret is not None:
                return return_vimp
        if "coef" in plots:
            coef_plot(
                self.fitted,
                alpha=alpha,
                intercept=intercept,
                incl=incl,
                excl=excl,
                figsize=figsize,
            )

    def f_test(self, test=None, dec=3) -> None:
        """
        F-test for competing models

        Parameters
        ----------
        test : list
            List of strings; contains the names of the columns of data to be tested
        """
        evar = setdiff(self.evar, test)
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

        print(f"\nModel 1: {form}")
        print(f"Model 2: {self.form}")
        out = self.fitted.f_test(hypothesis)

        r2_sub = self.fitted.rsquared - (
            len(hypothesis) * out.fvalue * (1 - self.fitted.rsquared)
        ) / (self.fitted.nobs - self.fitted.df_model - 1)

        pvalue = ifelse(out.pvalue < 0.001, "< .001", round(out.pvalue, dec))
        print(f"R-squared, Model 1 vs 2: {r2_sub:.3f} vs {self.fitted.rsquared:.3f}")
        print(
            f"F-statistic: {round(out.fvalue, dec)} df ({out.df_num:.0f}, {out.df_denom:.0f}), p.value {pvalue}"
        )
