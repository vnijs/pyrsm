import numpy as np
from scipy import stats

from .utils import iround, check, plot_continuous, ceil, floor


def prob_norm(mean, sd, lb=None, ub=None, plb=None, pub=None):
    if lb is not None:
        p_lb = stats.norm.cdf(lb, mean, sd)
    else:
        p_lb = None
    if ub is not None:
        p_ub = stats.norm.cdf(ub, mean, sd)
    else:
        p_ub = None
    if lb is not None and ub is not None:
        p_int = max(p_ub - p_lb, 0)
    else:
        p_int = None
    if plb is not None:
        v_lb = stats.norm.ppf(plb, mean, sd)
    else:
        v_lb = None
    if pub is not None:
        v_ub = stats.norm.ppf(pub, mean, sd)
    else:
        v_ub = None
    check(lb, ub, plb, pub)
    return {
        "p_lb": p_lb,
        "p_ub": p_ub,
        "p_int": p_int,
        "v_lb": v_lb,
        "v_ub": v_ub,
        "mean": mean,
        "sd": sd,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
    }


def plot_prob_norm(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]
    mean = dct["mean"]
    stdev = dct["sd"]
    x_range = np.linspace(floor(mean - 4 * stdev), ceil(mean + 4 * stdev), 1000)
    y_range = stats.norm.pdf(x_range, mean, stdev)
    return plot_continuous(x_range, y_range, lb, ub, title="Normal Distribution")


def summary_prob_norm(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    add(summary_dict, "Probability calculator", None)
    add(summary_dict, "Distribution", "Normal")
    mean, stdev = dct["mean"], dct["sd"]
    add(summary_dict, "Mean", round(mean, dec))
    add(summary_dict, "St. dev", round(stdev, dec))
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        add(summary_dict, "Lower bound", "-Inf" if lb is None else lb)
        add(summary_dict, "Upper bound", "Inf" if ub is None else ub)
        if ub is not None or lb is not None:
            if lb is not None:
                lb_r = round(lb, dec)
                add(summary_dict, f"P(X < {lb_r})", p_lb)
                add(summary_dict, f"P(X > {lb_r})", round(1 - p_lb, dec))
            if ub is not None:
                ub_r = round(ub, dec)
                add(summary_dict, f"P(X < {ub_r})", p_ub)
                add(summary_dict, f"P(X > {ub_r})", round(1 - p_ub, dec))
            if lb is not None and ub is not None:
                add(summary_dict, f"P({lb_r} < X < {ub_r})", p_int)
                add(summary_dict, f"1 - P({lb_r} < X < {ub_r})", round(1 - p_int, dec))
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]
        add(summary_dict, "Lower bound", "0" if plb is None or plb < 0 else plb)
        add(summary_dict, "Upper bound", "1" if pub is None or pub > 1 else pub)
        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            if plb is not None and plb >= 0:
                v_lb_r = round(v_lb, dec)
                add(summary_dict, f"P(X < {v_lb_r})", plb)
                add(summary_dict, f"P(X > {v_lb_r})", round(1 - plb, dec))
            if pub is not None and pub <= 1:
                v_ub_r = round(v_ub, dec)
                add(summary_dict, f"P(X < {v_ub_r})", pub)
                add(summary_dict, f"P(X > {v_ub_r})", round(1 - pub, dec))
            if (plb is not None and plb >= 0) and (pub is not None and pub <= 1):
                add(summary_dict, f"P({v_lb_r} < X < {v_ub_r})", round(pub - plb, dec))
                add(summary_dict, f"1 - P({v_lb_r} < X < {v_ub_r})", round(1 - (pub - plb), dec))
    if ret:
        return summary_dict
    else:
        for k, v in summary_dict.items():
            if v is not None:
                print(f"{k}: {v}")
