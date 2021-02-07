import pandas as pd
from scipy import stats


class cross_tabs:
    def __init__(self, df, var1, var2):
        self.df = df
        self.var1 = var1
        self.var2 = var2

        self.observed = pd.crosstab(
            df[var1], columns=df[var2], margins=True, margins_name="Total"
        )
        self.chisq = stats.chi2_contingency(
            self.observed.drop(columns="Total").drop("Total", axis=0), correction=False
        )
        expected = pd.DataFrame(self.chisq[3])
        expected["Total"] = expected.sum(axis=1)
        expected = expected.append(expected.sum(), ignore_index=True).set_index(
            self.observed.index
        )
        expected.columns = self.observed.columns
        self.expected = expected
        self.chi_sq = (
            ((self.observed - self.expected) ** 2 / self.expected)
            .drop(columns="Total")
            .drop("Total", axis=0)
        )
        self.perc_row = self.observed.div(self.observed["Total"], axis=0)
        self.perc_col = self.observed.div(self.observed.loc["Total", :], axis=1)
        self.perc = self.observed / self.observed.loc["Total", "Total"]

    def summary(self, output=["observed", "expected"], dec=2):
        prn = f"""
Cross-tabs
Data : titanic
Variables: survived, sex
Null hyp: there is no association between {self.var1} and {self.var2}
Alt. hyp: there is an association between {self.var1} and {self.var2}
"""
        if "observed" in output:
            prn = (
                prn
                + f"""
Observed:

{self.observed.applymap(lambda x: "{:,}".format(x))}
"""
            )
        if "observed" in output:
            prn = (
                prn
                + f"""
Expected: (row total x column total) / total

{self.expected.round(dec).applymap(lambda x: "{:,}".format(x))}
"""
            )
        if "chisq" in output:
            prn = (
                prn
                + f"""
Contribution to chi-squared: (o - e)^2 / e

{self.chi_sq.round(dec).applymap(lambda x: "{:,}".format(x))}
"""
            )
        if "perc_row" in output:
            prn = (
                prn
                + f"""
Row percentages:

{self.perc_row.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
            )
        if "perc_col" in output:
            prn = (
                prn
                + f"""
Column percentages:

{self.perc_col.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
            )

        if "perc_all" in output:
            prn = (
                prn
                + f"""
Percentages:

{self.perc.transform(lambda x: (100*x).round(dec).astype(str) + "%")}
"""
            )

        prn = (
            prn
            + f"""
Chi-squared: {round(self.chisq[0], dec)} df({round(self.chisq[2], dec)}), p.value {round(self.chisq[1], dec)}
"""
        )
        print(prn)

    # Another instance method
    def plot(self, output="perc_col"):
        pdf = getattr(self, output).drop(columns="Total").drop("Total", axis=0)
        fig = pdf.plot.bar()


def correlation(df, dec=3, prn=True):
    """
    Calculate correlations between the numeric variables in a Pandas dataframe

    Parameters
    ----------
    df : Pandas dataframe with numeric variables
    dec : int
        Number of decimal places to use in rounding
    prn : bool
        Print or return the correlation matrix

    Returns
    -------
    Pandas dataframe with all numeric variables standardized

    Examples
    --------
    df = pd.DataFrame({"x": [0, 1, 1, 1, 0], "y": [1, 0, 0, 0, np.NaN]})
    correlation(df)
    """
    df = df.copy()
    isNum = [pd.api.types.is_numeric_dtype(df[col]) for col in df.columns]
    isNum = list(compress(df.columns, isNum))
    df = df[isNum]

    ncol = df.shape[1]
    cr = np.zeros([ncol, ncol])
    cp = cr.copy()
    for i in range(ncol - 1):
        for j in range(i + 1, ncol):
            cdf = df.iloc[:, [i, j]]
            # pairwise deletion
            cdf = cdf[~np.any(np.isnan(cdf), axis=1)]
            c = stats.pearsonr(cdf.iloc[:, 0], cdf.iloc[:, 1])
            cr[j, i] = c[0]
            cp[j, i] = c[1]

    ind = np.triu_indices(ncol)

    # correlation matrix
    crs = cr.round(ncol).astype(str)
    crs[ind] = ""
    crs = pd.DataFrame(
        np.delete(np.delete(crs, 0, axis=0), crs.shape[1] - 1, axis=1),
        columns=df.columns[:-1],
        index=df.columns[1:],
    )

    # pvalues
    cps = cp.round(ncol).astype(str)
    cps[ind] = ""
    cps = pd.DataFrame(
        np.delete(np.delete(cps, 0, axis=0), cps.shape[1] - 1, axis=1),
        columns=df.columns[:-1],
        index=df.columns[1:],
    )

    if prn:
        print("Correlation matrix:")
        print(crs)
        print("\np.values:")
        print(cps)
    else:
        return cr, cp