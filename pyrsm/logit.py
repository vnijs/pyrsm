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
from .regression import sig_stars

# from statsmodels.regression.linear_model import RegressionResults as rrs
from .visualize import pred_plot_sm, vimp_plot_sm, extract_evars, extract_rvar


def get_mfit(fitted) -> tuple[Optional[dict], Optional[str]]:
    mfit_dct = None
    model_type = None
    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        fw = None
        if fitted.model._has_freq_weights:
            fw = fitted.model.freq_weights

        mfit_dct = {
            "pseudo_rsq_mcf": [1 - fitted.llf / fitted.llnull],
            "pseudo_rsq_mcf_adj": [1 - (fitted.llf - fitted.df_model) / fitted.llnull],
            "AUC": [auc(fitted.model.endog, fitted.fittedvalues, weights=fw)],
            "log_likelihood": fitted.llf,
            "AIC": [fitted.aic],
            "BIC": [fitted.bic_llf],
            "chisq": [fitted.pearson_chi2],
            "chisq_df": [fitted.df_model],
            "chisq_pval": [1 - stats.chi2.cdf(fitted.pearson_chi2, fitted.df_model)],
            "nobs": [fitted.nobs],
        }

        model_type = "logistic"

    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        mfit_dct = {
            "rsq": [fitted.rsquared],
            "rsq_adj": [fitted.rsquared_adj],
            "fvalue": [fitted.fvalue],
            "ftest_df_model": [fitted.df_model],
            "ftest_df_resid": [fitted.df_resid],
            "ftest_pval": [fitted.f_pvalue],
            "nobs": [fitted.nobs],
        }

        model_type = "regression"

    return mfit_dct, model_type


def or_ci(fitted, alpha=0.05, intercept=False, importance=False, data=None, dec=3):
    """
    Confidence interval for Odds ratios

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in output (True or False)
    importance : int
        Calculate variable importance. Only meaningful if data
        used in estimation was standardized prior to model
        estimation
    data : Pandas dataframe
        Unstandardized data used to calculate descriptive
        statistics
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe with Odd-ratios and confidence intervals
    """

    df = pd.DataFrame(np.exp(fitted.params), columns=["OR"]).dropna()
    df["OR%"] = 100 * ifelse(df["OR"] < 1, -(1 - df["OR"]), df["OR"] - 1)

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    df[[f"{low}%", f"{high}%"]] = np.exp(fitted.conf_int(alpha=alpha))
    df["p.values"] = ifelse(fitted.pvalues < 0.001, "< .001", fitted.pvalues.round(dec))
    df["  "] = sig_stars(fitted.pvalues)
    df["OR%"] = [f"{round(o, max(dec-2, 0))}%" for o in df["OR%"]]
    df = df.reset_index()

    if importance:
        df["dummy"] = df["index"].str.contains("[T", regex=False)
        df["importance"] = (
            pd.DataFrame().assign(OR=df["OR"], ORinv=1 / df["OR"]).max(axis=1)
        )

    if isinstance(data, pd.DataFrame):
        # using a fake response variable variable
        data = data.assign(__rvar__=1).copy()
        form = "__rvar__ ~ " + fitted.model.formula.split("~", 1)[1]
        exog = pd.DataFrame(smf.logit(formula=form, data=data).exog)
        weights = fitted._freq_weights
        if sum(weights) > len(weights):

            def wmean(x):
                return weighted_mean(x, weights)

            def wstd(x):
                return weighted_sd(pd.DataFrame(x), weights)[0]

            df = pd.concat(
                [df, exog.apply([wmean, wstd, "min", "max"]).T],
                axis=1,
            )
        else:
            df = pd.concat([df, exog.apply(["mean", "std", "min", "max"]).T], axis=1)

    if intercept is False:
        df = df.loc[df["index"] != "Intercept"]

    return df.round(dec)


def or_plot(fitted, alpha=0.05, intercept=False, incl=None, excl=None, figsize=None):
    """
    Odds ratio plot

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in odds-ratio plot (True or False)
    incl : str or list of strings
        Variables to include in the odds-ratio plot. All will be included by default
    excl : str or list of strings
        Variables to exclude from the odds-ratio plot. None are excluded by default

    Returns
    -------
    Matplotlit object
        Plot of Odds ratios
    """

    # iloc to reverse order
    df = or_ci(fitted, alpha=alpha, intercept=intercept, dec=100).dropna().iloc[::-1]

    if incl is not None:
        incl = ifelse(isinstance(incl, str), [incl], incl)
        rx = "(" + "|".join([f"^{v}$|^{v}\\[" for v in incl]) + ")"
        incl = df["index"].str.match(rf"{rx}")
        if intercept:
            incl[0] = True
        df = df[incl]

    if excl is not None:
        excl = ifelse(isinstance(excl, str), [excl], excl)
        rx = "(" + "|".join([f"^{v}$|^{v}\\[" for v in excl]) + ")"
        excl = df["index"].str.match(rf"{rx}")
        if intercept:
            excl[0] = False
        df = df[~excl]

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    err = [df["OR"] - df[f"{low}%"], df[f"{high}%"] - df["OR"]]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.axvline(1, ls="dashdot")
    ax.errorbar(x="OR", y="index", data=df, xerr=err, fmt="none")
    ax.scatter(x="OR", y="index", data=df)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_locator(ticker.LogLocator(subs=[0.1, 0.2, 0.5, 1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.set(xlabel="Odds-ratio")
    return ax


def vif(fitted, dec=3):
    """
    Calculate the Variance Inflation Factor (VIF) associated with each
    exogenous variable

    Status
    ------
    WIP port of VIF calculation from R's car:::vif.default to Python

    Parameters
    ----------
    fitted : A fitted (logistic) regression model
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe sorted by VIF score
    """

    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        # legacy for when only an un-fitted model was accepted
        model = fitted

    vif = [variance_inflation_factor(model.exog, i) for i in range(model.exog.shape[1])]
    df = pd.DataFrame(model.exog_names, columns=["variable"])
    df["vif"] = vif
    df["Rsq"] = 1 - 1 / df["vif"]

    if "Intercept" in model.exog_names:
        df = df[df["variable"] != "Intercept"]

    df = df.sort_values("vif", ascending=False).reset_index(drop=True)

    if dec is not None:
        df = df.round(dec)

    return df


def predict_ci(fitted, df, alpha=0.05):
    """
    Compute predicted probabilities with confidence intervals based on a
    logistic regression model

    Parameters
    ----------
    fitted : Logistic regression model fitted using the statsmodels formula interface
    df : Pandas dataframe with input data for prediction
    alpha : float
        Significance level (0-1). Default is 0.05

    Returns
    -------
    Pandas dataframe with probability predictions and lower and upper confidence bounds

    Example
    -------
    import numpy as np
    import statsmodels.formula.api as smf
    import pandas as pd

    # simulate data
    np.random.seed(1)
    x1 = np.arange(100)
    x2 = pd.Series(["a", "b", "c", "a"], dtype="category").sample(100, replace=True)
    y = (x1 * 0.5 + np.random.normal(size=100, scale=10) > 30).astype(int)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    # estimate the model
    model = smf.logit(formula="y ~ x1 + x2", data=df).fit()
    model.summary()
    pred = predict_ci(model, df)

    plt.clf()
    plt.plot(x1, pred["prediction"])
    plt.plot(x1, pred["2.5%"], color='black', linestyle="--", linewidth=0.5)
    plt.plot(x1, pred["97.5%"], color='black', linestyle="--", linewidth=0.5)
    plt.show()
    """

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be a numeric value between 0 and 1")

    # generate predictions
    prediction = fitted.predict(df)

    # set up the data in df in the same was as the exog data
    # that is part of fitted.model.exog
    # use a fake response variable
    df = df.assign(__rvar__=1).copy()
    form = "__rvar__ ~ " + fitted.model.formula.split("~", 1)[1]
    exog = smf.logit(formula=form, data=df).exog

    low, high = [alpha / 2, 1 - (alpha / 2)]
    Xb = np.dot(exog, fitted.params)
    se = np.sqrt((exog.dot(fitted.cov_params()) * exog).sum(-1))
    me = stats.norm.ppf(high) * se

    return pd.DataFrame(
        {
            "prediction": prediction,
            f"{low*100}%": expit(Xb - me),
            f"{high*100}%": expit(Xb + me),
        }
    )


def model_fit(fitted, dec: int = 3, prn: bool = True) -> Union[str, pd.DataFrame]:
    """
    Compute various model fit statistics for a fitted linear or logistic regression model

    Parameters
    ----------
    fitted : statmodels ols or glm object
        Regression model fitted using statsmodels
    dec : int
        Number of decimal places to use in rounding
    prn : bool
        If True, print output, else return a Pandas dataframe with the results

    Returns
    -------
        If prn is True, print output, else return a Pandas dataframe with the results
    """
    if hasattr(fitted, "df_resid"):
        weighted_nobs = (
            fitted.df_resid + fitted.df_model + int(hasattr(fitted.params, "Intercept"))
        )
        if weighted_nobs > fitted.nobs:
            fitted.nobs = weighted_nobs
    mfit_dct, model_type = get_mfit(fitted)
    if not mfit_dct:
        return "Only linear and logistic regression models are currently supported."

    mfit = pd.DataFrame(mfit_dct)

    if prn:
        output = "Model type not supported"
        if model_type == "logistic":
            output = f"""Pseudo R-squared (McFadden): {mfit.pseudo_rsq_mcf.values[0].round(dec)}
Pseudo R-squared (McFadden adjusted): {mfit.pseudo_rsq_mcf_adj.values[0].round(dec)}
Area under the RO Curve (AUC): {mfit.AUC.values[0].round(dec)}
Log-likelihood: {mfit.log_likelihood.values[0].round(dec)}, AIC: {mfit.AIC.values[0].round(dec)}, BIC: {mfit.BIC.values[0].round(dec)}
Chi-squared: {mfit.chisq.values[0].round(dec)}, df({mfit.chisq_df.values[0]}), p.value {np.where(mfit.chisq_pval.values[0] < .001, "< 0.001", mfit.chisq_pval.values[0].round(dec))} 
Nr obs: {mfit.nobs.values[0]:,.0f}"""
        elif model_type == "regression":
            output = f"""R-squared: {mfit.rsq.values[0].round(dec)}, Adjusted R-squared: {mfit.rsq_adj.values[0].round(dec)}
F-statistic: {mfit.fvalue[0].round(dec)} df({mfit.ftest_df_model.values[0]:.0f}, {mfit.ftest_df_resid.values[0]:.0f}), p.value {np.where(mfit.ftest_pval.values[0] < .001, "< 0.001", mfit.ftest_pval.values[0].round(dec))}
Nr obs: {mfit.nobs.values[0]:,.0f}"""
        return output
    else:
        return mfit


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
        rvar: String; name of the column to be used as the response variable
        lev: String; name of the level in the response variable
        ssq: Boolean; whether sum of squared errors need to be reported
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

        self.coef["OR"] = self.coef["OR"].round(dec)
        self.coef["coefficient"] = self.coef["coefficient"].round(2)
        self.coef["std.error"] = self.coef["std.error"].round(dec)
        self.coef["z.value"] = self.coef["z.value"].round(dec)
        self.coef["p.value"] = ifelse(
            self.coef["p.value"] < 0.001, "< .001", self.coef["p.value"].round(dec)
        )
        self.coef["OR%"] = [f"{round(o, max(dec-2, 0))}%" for o in self.coef["OR%"]]
        print(f"\n{self.coef.to_string(index=False)}")
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
