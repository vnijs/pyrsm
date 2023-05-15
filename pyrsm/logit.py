from typing import Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.special import expit
from .utils import ifelse
from .stats import weighted_mean, weighted_sd
from .perf import auc
from .model import sig_stars, model_fit, or_ci, or_plot, sim_prediction

# from statsmodels.regression.linear_model import RegressionResults as rrs
from .visualize import pred_plot_sm, vimp_plot_sm, extract_evars, extract_rvar


class logistic:
    def __init__(
        self,
        dataset: pd.DataFrame,
        rvar: Optional[str] = None,
        lev: Optional[str] = None,
        evar: Optional[list[str]] = None,
        form: Optional[str] = None,
    ) -> None:
        """
        Initialize logistic regression model

        Parameters
        ----------
        dataset: pandas DataFrame; dataset
        evar: List of strings; contains the names of the columns of data to be used as explanatory variables
        lev: String; name of the level in the response variable
        rvar: String; name of the column to be used as the response variable
        form: String; formula for the regression equation to use if evar and rvar are not provided
        """
        self.dataset = dataset
        self.rvar = rvar
        self.lev = lev
        self.evar = evar
        self.form = form

        if self.form:
            self.fitted = smf.glm(
                formula=self.form, data=self.dataset, family=Binomial(link=logit())
            ).fit()
            self.evar = extract_evars(self.fitted.model, self.dataset.columns)
            self.rvar = extract_rvar(self.fitted.model, self.dataset.columns)
            self.lev = self.dataset.at[0, self.rvar]
        else:
            self.form = f"{self.rvar} ~ {' + '.join(self.evar)}"
            self.fitted = smf.glm(
                formula=self.form, data=self.dataset, family=Binomial(link=logit())
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

    def summary(self, dec=3) -> None:
        """
        Summarize output from a logistic regression model
        """

        if hasattr(self.dataset, "description"):
            data_name = self.dataset.description.split("\n")[0].split()[1].lower()
        else:
            data_name = "Not available"

        print("Logistic regression (GLM)")
        print(f"Data                 : {data_name}")
        print(f"Response variable    : {self.rvar}")
        print(f"Level                : {self.lev}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        print(f"Null hyp.: there is no effect x on {self.rvar}")
        print(f"Alt. hyp.: there is an effect of x on {self.rvar}")

        df = self.coef.copy()
        df["OR"] = df["OR"].round(dec)
        df["coefficient"] = df["coefficient"].round(2)
        df["std.error"] = df["std.error"].round(dec)
        df["z.value"] = df["z.value"].round(dec)
        df["p.value"] = ifelse(
            df["p.value"] < 0.001, "< .001", df["p.value"].round(dec)
        )
        df["OR%"] = [f"{round(o, max(dec-2, 0))}%" for o in df["OR%"]]
        print(f"\n{df.to_string(index=False)}")
        print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print(f"\n{model_fit(self.fitted)}")

    def ci(self, alpha=0.05, intercept=False, dec=3) -> None:
        """
        Confidence intervals for Odds Ratios
        """
        print(
            f"\n{or_ci(self.fitted, alpha=alpha, intercept=intercept).to_string(index=False)}"
        )

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
                self.dataset,
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
            vimp_plot_sm(self.fitted, self.dataset, rep=10, ax=None, ret=False)
