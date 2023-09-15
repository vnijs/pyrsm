from typing import Union, Optional
import pandas as pd
import numpy as np
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
import statsmodels.formula.api as smf
from scipy import stats
from pyrsm.utils import ifelse, setdiff
from pyrsm.model.model import (
    sig_stars,
    model_fit,
    or_ci,
    or_plot,
    sim_prediction,
    predict_ci,
)
from pyrsm.model.model import vif as calc_vif

# from statsmodels.regression.linear_model import RegressionResults as rrs
from .visualize import pred_plot_sm, vimp_plot_sm, extract_evars, extract_rvar


class logistic:
    def __init__(
        self,
        data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
        rvar: Optional[str] = None,
        lev: Optional[str] = None,
        evar: Optional[list[str]] = None,
        ivar: Optional[list[str]] = None,
        form: Optional[str] = None,
    ) -> None:
        """
        Initialize logistic regression model

        Parameters
        ----------
        data: pandas DataFrame; dataset
        evar: List of strings; contains the names of the columns of data to be used as explanatory variables
        lev: String; name of the level in the response variable
        rvar: String; name of the column to be used as the response variable
        form: String; formula for the regression equation to use if evar and rvar are not provided
        """
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name].copy()  # needed with pandas
        else:
            self.data = data.copy()  # needed with pandas
            self.name = "Not provided"
        self.rvar = rvar
        self.lev = lev
        self.evar = ifelse(isinstance(evar, str), [evar], evar)
        self.ivar = ifelse(isinstance(ivar, str), [ivar], ivar)
        self.form = form

        if self.lev is not None and self.rvar is not None:
            self.data[self.rvar] = (self.data[self.rvar] == lev).astype(int)

        if self.form:
            self.fitted = smf.glm(
                formula=self.form, data=self.data, family=Binomial(link=Logit())
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
                formula=self.form, data=self.data, family=Binomial(link=Logit())
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

    def summary(self, ci=False, vif=False, test=None, dec=3) -> None:
        """
        Summarize output from a logistic regression model
        """
        print("Logistic regression (GLM)")
        print(f"Data                 : {self.name}")
        print(f"Response variable    : {self.rvar}")
        print(f"Level                : {self.lev}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
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
        print(f"\n{model_fit(self.fitted)}")

        if ci:
            print("\nConfidence intervals:")
            df = or_ci(self.fitted).set_index("index")
            df.index.name = None
            print(f"\n{df.to_string()}")

        if vif:
            if self.evar is None or len(self.evar) < 2:
                print("\nVariance Inflation Factors cannot be calculated")
            else:
                print("\nVariance inflation factors:")
                print(f"\n{calc_vif(self.fitted).to_string()}")

        if test is not None and len(test) > 0:
            self.chisq_test(test=test, dec=dec)

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
            pred = pd.DataFrame().assign(prediction=self.fitted.predict(data))
            return pd.concat([data, pred], axis=1)

    def plot(
        self,
        plots="or",
        alpha=0.05,
        intercept=False,
        incl=None,
        excl=None,
        incl_int=[],
        fix=True,
        hline=False,
        figsize=None,
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
                self.data,
                incl=incl,
                excl=[],
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=20,
                minq=0.025,
                maxq=0.975,
            )
        if "vimp" in plots:
            vimp_plot_sm(self.fitted, self.data, rep=10, ax=None, ret=False)

    def chisq_test(self, test=None, dec=3) -> None:
        """
        Chisq-test for competing models

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
            formula=form, data=self.data, family=Binomial(link=Logit())
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
