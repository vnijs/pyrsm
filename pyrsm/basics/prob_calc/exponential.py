import numpy as np
from scipy import stats
from .utils import iround, check, make_colors_continuous


def prob_expo(rate, lb=None, ub=None, plb=None, pub=None):
    if lb is not None and lb < 0:
        lb = 0
    if ub is not None and ub < 0:
        ub = 0
    if lb is None:
        p_lb = None
    else:
        p_lb = stats.expon.cdf(lb, scale=1 / rate)
    if ub is None:
        p_ub = None
    else:
        p_ub = stats.expon.cdf(ub, scale=1 / rate)
    p_int = None
    if lb is not None and ub is not None:
        if lb > ub:
            raise ValueError("Please ensure the lower bound is smaller than the upper bound value")
        else:
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
        v_lb = stats.expon.ppf(plb, scale=1 / rate)
    else:
        v_lb = None
    if pub is not None:
        v_ub = stats.expon.ppf(pub, scale=1 / rate)
    else:
        v_ub = None
    check(lb, ub, plb, pub)
    return {
        "p_lb": p_lb,
        "p_ub": p_ub,
        "p_int": p_int,
        "v_lb": v_lb,
        "v_ub": v_ub,
        "rate": rate,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
    }


def summary_prob_expo(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    add(summary_dict, "Distribution", "Exponential")
    rate = dct["rate"]
    add(summary_dict, "Rate", rate)
    add(summary_dict, "Mean", round(1 / rate, dec))
    add(summary_dict, "Variance", round(rate**-2, dec))
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
                add(summary_dict, f"P({v_lb} < X < {v_ub})", pub - plb)
                add(summary_dict, f"1 - P({v_lb} < X < {v_ub})", round(1 - (pub - plb), dec))

    if ret:
        return summary_dict
    else:
        from .utils import pretty_print_summary

        pretty_print_summary(summary_dict)


def plot_prob_expo(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]
    rate = dct["rate"]
    x_range = np.linspace(0, stats.expon.ppf(0.99, scale=1 / rate), 1000)
    y_range = stats.expon.pdf(x_range, scale=1 / rate)
    make_colors_continuous(ub, lb, x_range, y_range)
