import pandas as pd
from typing import Optional
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults as rrs
from utils import setdiff
import numpy as np


class logistic_regression:
    def __init__(
        self,
        dataset: pd.DataFrame,
        rvar: Optional[str] = None,
        rvar_level: Optional[str] = None,
        evars: Optional[list[str]] = None,
        form: Optional[str] = None,
    ) -> None:
        """
        Initialize logistic regression model

        Parameters
        ----------
        dataset: pandas DataFrame; dataset
        evars: List of strings; contains the names of the columns of data to be used as explanatory variables
        rvar: String; name of the column to be used as the response variable
        rvar_level: String; name of the level in the response variable
        ssq: Boolean; whether sum of squared errors need to be reported
        form: String; formula for the regression equation to use if evars and rvar are not provided
        """
        self.dataset = dataset
        self.rvar = rvar
        self.rvar_level = rvar_level
        self.evars = evars
        self.form = form

        if self.form:
            self.model = smf.glm(formula=self.form, data=self.dataset)
            self.evars = [
                np.setdiff1d(
                    np.unique(np.array(self.model.exog_names)),
                    np.unique(np.array("const")),
                    assume_unique=True,
                )
            ]
            self.evars = setdiff(self.model.exog_names, "const")
            self.rvar = self.model.endog_names
            self.rvar_level = self.dataset.at[0, self.rvar]
        else:
            self.form = f"{self.rvar} ~ {' + '.join(self.evars)}"
            print(self.form)
            self.model = smf.glm(
                formula=self.form, data=self.dataset, family=sm.families.Binomial()
            )

    def regress(self) -> rrs:
        """
        Estimate logistic regression model

        Returns
        -------
        res: Object with fitted values and residuals
        """
        res = self.model.fit()

        data_name = ""
        if hasattr(self.dataset, "description"):
            data_name = self.dataset.description.split("\n")[0].split()[1].lower()

        print(f"Data: {data_name}")
        print(f"Response variable: {self.rvar}")
        evars_str = ", ".join(self.evars)
        print(f"Explanatory variables: {evars_str}")
        print(f"Null hyp.: the effect of x on {self.rvar} is zero")
        print(f"Alt. hyp.: the effect of x on {self.rvar} is not zero")

        summary = res.summary()
        print(f"\n {summary}")

        return res
