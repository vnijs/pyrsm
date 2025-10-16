import numpy as np
from scipy import stats
from .utils import iround, check, make_colors_continuous


def prob_fdist(df1, df2, lb=None, ub=None, plb=None, pub=None):
    if lb is not None and lb < 0:
        lb = 0
    if ub is not None and ub < 0:
        ub = 0
    if lb is None:
        p_lb = None
    else:
        p_lb = stats.f.cdf(lb, df1, df2)
    if ub is None:
        p_ub = None
    else:
        p_ub = stats.f.cdf(ub, df1, df2)
    p_int = None
    if lb is not None and ub is not None:
        p_int = max(p_ub - p_lb, 0)
    if pub is not None:
        if pub < 0:
            pub = 0
        elif pub > 1:
            pub = 1
    if plb is not None:
        if plb < 0:
            plb = 0
        elif plb > 1:
            plb = 1
    if plb is not None:
        v_lb = stats.f.ppf(plb, df1, df2)
    else:
        v_lb = None
    if pub is not None:
        v_ub = stats.f.ppf(pub, df1, df2)
    else:
        v_ub = None
    check(lb, ub, plb, pub)
    return {
        "p_lb": p_lb,
        "p_ub": p_ub,
        "p_int": p_int,
        "v_lb": v_lb,
        "v_ub": v_ub,
        "df1": df1,
        "df2": df2,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
    }


def summary_prob_fdist(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    add(summary_dict, "Distribution", "F")
    df1, df2 = dct["df1"], dct["df2"]
    add(summary_dict, "Df 1", df1)
    add(summary_dict, "Df 2", df2)
    m = round(df2 / (df2 - 2), dec) if df2 > 2 else "NA"
    variance = (
        round((2 * df2**2 * (df1 + df2 - 2)) / (df1 * (df2 - 2) ** 2 * (df2 - 4)), dec)
        if df2 > 4
        else np.nan
    )
    add(summary_dict, "Mean", m)
    add(summary_dict, "Variance", variance)
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        add(summary_dict, "Lower bound", "0" if lb is None else lb)
        add(summary_dict, "Upper bound", "Inf" if ub is None else ub)
        if ub is not None or lb is not None:
            if lb is not None:
                lb = round(lb, dec)
                add(summary_dict, f"P(X < {lb})", p_lb)
                add(summary_dict, f"P(X > {lb})", round(1 - p_lb, dec))
            if ub is not None:
                ub = round(ub, dec)
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
                v_lb = round(v_lb, dec)
                add(summary_dict, f"P(X < {v_lb})", plb)
                add(summary_dict, f"P(X > {v_lb})", round(1 - plb, dec))
            if pub is not None and pub <= 1:
                v_ub = round(v_ub, dec)
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


def plot_prob_fdist(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]
    df1, df2 = dct["df1"], dct["df2"]
    x_range = np.linspace(0, stats.f.ppf(0.99, df1, df2), 1000)
    y_range = stats.f.pdf(x_range, df1, df2)
    make_colors_continuous(ub, lb, x_range, y_range)
