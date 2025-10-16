import numpy as np
from scipy import stats
from .utils import iround, check, make_colors_continuous, ceil, floor


def prob_chisq(df, lb=None, ub=None, plb=None, pub=None):
    if lb is not None:
        p_lb = stats.chi2.cdf(lb, df)
    else:
        p_lb = None
    if ub is not None:
        p_ub = stats.chi2.cdf(ub, df)
    else:
        p_ub = None
    if lb is not None and ub is not None:
        p_int = max(p_ub - p_lb, 0)
    else:
        p_int = None
    if plb is not None:
        v_lb = stats.chi2.ppf(plb, df)
    else:
        v_lb = None
    if pub is not None:
        v_ub = stats.chi2.ppf(pub, df)
    else:
        v_ub = None
    check(lb, ub, plb, pub)
    return {
        "p_lb": p_lb,
        "p_ub": p_ub,
        "p_int": p_int,
        "v_lb": v_lb,
        "v_ub": v_ub,
        "df": df,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
    }


def plot_prob_chisq(dct, type="values"):
    import matplotlib.pyplot as plt

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]
    df = dct["df"]
    x_range = np.linspace(
        floor(stats.chi2.ppf(0.001, df)), ceil(stats.chi2.ppf(1 - 0.001, df)), 1000
    )
    y_range = stats.chi2.pdf(x_range, df)
    make_colors_continuous(ub, lb, x_range, y_range)


def summary_prob_chisq(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    add(summary_dict, "Distribution", "Chi-square")
    df = dct["df"]
    add(summary_dict, "Df", df)
    add(summary_dict, "Mean", df)
    add(summary_dict, "Variance", 2 * df)
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        add(summary_dict, "Lower bound", "0" if lb is None else lb)
        add(summary_dict, "Upper bound", "Inf" if ub is None else ub)
        if ub is not None or lb is not None:
            if lb is not None:
                add(summary_dict, f"P(X < {lb})", p_lb)
                add(summary_dict, f"P(X > {lb})", round(1 - p_lb, dec))
            if ub is not None:
                add(summary_dict, f"P(X < {ub})", p_ub)
                add(summary_dict, f"P(X > {ub})", round(1 - p_ub, dec))
            if lb is not None and ub is not None:
                add(summary_dict, f"P({lb} < X < {ub})", p_int)
                add(summary_dict, f"1 - P({lb} < X < {ub})", round(1 - p_int, dec))
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]
        add(summary_dict, "Lower bound", "0" if plb is None or plb < 0 else plb)
        add(summary_dict, "Upper bound", "1" if pub is None or pub > 1 else pub)
        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            if plb is not None and plb >= 0:
                add(summary_dict, f"P(X < {v_lb})", plb)
                add(summary_dict, f"P(X > {v_lb})", round(1 - plb, dec))
            if pub is not None and pub <= 1:
                add(summary_dict, f"P(X < {v_ub})", pub)
                add(summary_dict, f"P(X > {v_ub})", round(1 - pub, dec))
            if (plb is not None and plb >= 0) and (pub is not None and pub <= 1):
                add(summary_dict, f"P({v_lb} < X < {v_ub})", round(pub - plb, dec))
                add(summary_dict, f"1 - P({v_lb} < X < {v_ub})", round(1 - (pub - plb), dec))

    if ret:
        return summary_dict
    else:
        from .utils import pretty_print_summary

        pretty_print_summary(summary_dict)
