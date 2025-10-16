import numpy as np
from scipy import stats
from .utils import iround, check, make_colors_continuous


def prob_unif(min, max, lb=None, ub=None, plb=None, pub=None):
    if min > max:
        raise ValueError("The maximum value must be larger than the minimum value")
    if lb is not None:
        p_lb = stats.uniform.cdf(lb, min, max - min)
    else:
        p_lb = None
    if ub is not None:
        p_ub = stats.uniform.cdf(ub, min, max - min)
    else:
        p_ub = None
    if lb is not None and ub is not None:
        p_int = np.max(p_ub - p_lb, 0)
    else:
        p_int = None
    if plb is not None:
        v_lb = stats.uniform.ppf(plb, min, max - min)
    else:
        v_lb = None
    if pub is not None:
        v_ub = stats.uniform.ppf(pub, min, max - min)
    else:
        v_ub = None
    mean = (max + min) / 2
    stdev = np.sqrt((max - min) ** 2 / 12)
    check(lb, ub, plb, pub)
    return {
        "p_lb": p_lb,
        "p_ub": p_ub,
        "p_int": p_int,
        "v_lb": v_lb,
        "v_ub": v_ub,
        "mean": mean,
        "stdev": stdev,
        "min": min,
        "max": max,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
    }


def summary_prob_unif(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    add(summary_dict, "Distribution", "Uniform")
    min, max = dct["min"], dct["max"]
    mean, stdev = dct["mean"], dct["stdev"]
    add(summary_dict, "Min", min)
    add(summary_dict, "Max", max)
    add(summary_dict, "Mean", round(mean, dec))
    add(summary_dict, "St. dev", round(stdev, dec))
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        add(summary_dict, "Lower bound", min if lb is None else lb)
        add(summary_dict, "Upper bound", max if ub is None else ub)
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


def plot_prob_unif(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]
    min, max = dct["min"], dct["max"]
    x_range = np.linspace(min, max, 1000)
    y_range = stats.uniform.pdf(x_range, min, max - min)
    ax = make_colors_continuous(ub, lb, x_range, y_range)
    ax.axvline(min, ymin=0.048, ymax=0.952, color="black", linewidth=1)
    ax.axvline(max, ymin=0.048, ymax=0.952, color="black", linewidth=1)
