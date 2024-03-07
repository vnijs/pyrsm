from typing import Union, Optional
import pandas as pd
import polars as pl
import numpy as np
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
import statsmodels.formula.api as smf
from scipy import stats
from pyrsm.utils import ifelse, setdiff, check_dataframe
from pyrsm.model.model import (
    sig_stars,
    model_fit,
    or_ci,
    or_plot,
    sim_prediction,
    predict_ci,
    convert_binary,
    convert_to_list,
)
from pyrsm.model.model import vif as calc_vif
from .visualize import pred_plot_sm, vimp_plot_sm, extract_evars, extract_rvar


class logistic:
    """
    Initialize logistic regression model

    Parameters
    ----------
    data: pandas DataFrame; dataset
    rvar: String; name of the column to be used as the response variable
    lev: String; name of the level in the response variable
    evar: List of strings; contains the names of the columns of data to be used as explanatory variables
    ivar: List of strings; contains the names of columns to interact and add as explanatory variables (e.g., ["x1:x2", "x3:x4])
    form: String; formula for the regression equation to use if evar and rvar are not provided
    weights: String; name of the column that contains frequency weights
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
            self.data = convert_binary(self.data, self.rvar, self.lev)

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
        df = pd.DataFrame(np.exp(self.fitted.params), columns=["OR"]).dropna()
        df["OR%"] = 100 * ifelse(df["OR"] < 1, -(1 - df["OR"]), df["OR"] - 1)
        df["coefficient"] = self.fitted.params
        df["std.error"] = self.fitted.params / self.fitted.tvalues
        # wierd but this is what statsmodels uses in summary
        df["z.value"] = self.fitted.tvalues
        df["p.value"] = self.fitted.pvalues
        df["  "] = sig_stars(self.fitted.pvalues)
        self.coef = df.reset_index()

    def summary(
        self, main=True, fit=True, ci=False, vif=False, test=None, dec=3
    ) -> None:
        """
        Summarize output from a logistic regression model

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
            df["coefficient"] = df["coefficient"].round(2)
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
        Predict probabilities for a logistic regression model
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
        plots="or",
        data=None,
        alpha=0.05,
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
        """
        if "or" in plots:
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
                data=ifelse(data is None, self.data[self.evar + [self.rvar]], data),
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

    def chisq_test(self, test=None, dec=3) -> None:
        """
        Chisq-test for competing models

        Parameters
        ----------
        test : list
            List of strings; contains the names of the columns of data to be tested
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

        # ensure constraints are unique
        # pattern = r"(\[T\.[^\]]*\])\:"
        # hypotheses = list(
        #     set(
        #         [
        #             f"({c} = 0)"
        #             for c in self.fitted.model.exog_names
        #             for v in test
        #             if f"{v}:" in c
        #             or f":{v}" in c
        #             or f"{v}[T." in c
        #             or v == c
        #             or v == re.sub(pattern, ":", c)
        #         ]
        #     )
        # )

        print(f"\nModel 1: {form}")
        print(f"Model 2: {self.form}")

        # Wald test (faster but not as accurate)
        # out = self.fitted.wald_test(hypotheses, scalar=True)
        # pvalue = ifelse(out.pvalue < 0.001, "< .001", round(out.pvalue, dec))
        # print(
        #     f"Chi-squared: {round(out.statistic, dec)} df ({out.df_denom:.0f}), p.value {pvalue}"
        # )

        # LR test of competing models (slower but more accurate)
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
