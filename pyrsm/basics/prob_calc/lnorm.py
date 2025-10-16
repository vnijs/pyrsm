import numpy as np
from scipy import stats
from .utils import iround, check, make_colors_continuous


def prob_lnorm(meanlog, sdlog, lb=None, ub=None, plb=None, pub=None):
    if lb is None:
        lb = -np.inf
    if ub is None:
        ub = np.inf
    p_ub = stats.lognorm.cdf(ub, sdlog, scale=np.exp(meanlog))
    p_lb = stats.lognorm.cdf(lb, sdlog, scale=np.exp(meanlog))
    p_int = max(p_ub - p_lb, 0)
    if pub is not None:
        if pub > 1:
            pub = 1
        if pub < 0:
            pub = 0
    if plb is not None:
        if plb > 1:
            plb = 1
        if plb < 0:
            plb = 0
    v_ub = stats.lognorm.ppf(pub, sdlog, scale=np.exp(meanlog)) if pub is not None else None
    v_lb = stats.lognorm.ppf(plb, sdlog, scale=np.exp(meanlog)) if plb is not None else None
    check(lb, ub, plb, pub)
    return {
        "meanlog": meanlog,
        "sdlog": sdlog,
        "lb": lb,
        "ub": ub,
        "p_lb": p_lb,
        "p_ub": p_ub,
        "p_int": p_int,
        "plb": plb,
        "pub": pub,
        "v_lb": v_lb,
        "v_ub": v_ub,
    }


def summary_prob_lnorm(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    meanlog, sdlog = dct["meanlog"], dct["sdlog"]
    lb, v_lb = dct["lb"], dct["v_lb"]
    ub, v_ub = dct["ub"], dct["v_ub"]
    p_lb, plb = dct["p_lb"], dct["plb"]
    p_ub, pub = dct["p_ub"], dct["pub"]
    p_int = dct["p_int"]

    add(summary_dict, "Distribution", "Log Normal")
    add(summary_dict, "Mean log", round(meanlog, dec))
    add(summary_dict, "St. dev log", round(sdlog, dec))

    if type == "values":
        add(summary_dict, "Lower bound", lb if lb is not None else "-Inf")
        add(summary_dict, "Upper bound", ub if ub is not None else "Inf")
        if ub != np.inf and lb != -np.inf:
            add(summary_dict, f"P(X < {lb if lb is not None else '-Inf'})", p_lb)
            add(summary_dict, f"P(X > {lb if lb is not None else '-Inf'})", round(1 - p_lb, dec))
            add(summary_dict, f"P(X < {ub if ub is not None else 'Inf'})", p_ub)
            add(summary_dict, f"P(X > {ub if ub is not None else 'Inf'})", round(1 - p_ub, dec))
            add(
                summary_dict,
                f"P({lb if lb is not None else '-Inf'} < X < {ub if ub is not None else 'Inf'})",
                round(p_int, dec),
            )
            add(
                summary_dict,
                f"1 - P({lb if lb is not None else '-Inf'} < X < {ub if ub is not None else 'Inf'})",
                round(1 - p_int, dec),
            )
        elif lb != -np.inf:
            add(summary_dict, f"P(X < {lb if lb is not None else '-Inf'})", p_lb)
            add(summary_dict, f"P(X > {lb if lb is not None else '-Inf'})", round(1 - p_lb, dec))
        elif ub != np.inf:
            add(summary_dict, f"P(X < {ub if ub is not None else 'Inf'})", p_ub)
            add(summary_dict, f"P(X > {ub if ub is not None else 'Inf'})", round(1 - p_ub, dec))
    else:
        if pub is None:
            pub = 2
        if plb is None:
            plb = -1
        add(summary_dict, "Lower bound", ("0" if plb < 0 else plb) + " (" + str(v_lb) + ")")
        add(summary_dict, "Upper bound", ("1" if pub > 1 else pub) + " (" + str(v_ub) + ")")
        if pub <= 1 or plb >= 0:
            if plb >= 0:
                add(summary_dict, f"P(X < {v_lb})", plb)
                add(summary_dict, f"P(X > {v_lb})", round(1 - plb, dec))
            if pub <= 1:
                add(summary_dict, f"P(X < {v_ub})", pub)
                add(summary_dict, f"P(X > {v_ub})", round(1 - pub, dec))
            if pub <= 1 and plb >= 0:
                add(summary_dict, f"P({v_lb} < X < {v_ub})", round(pub - plb, dec))
                add(summary_dict, f"1 - P({v_lb} < X < {v_ub})", round(1 - (pub - plb), dec))

    if ret:
        return summary_dict
    else:
        from .utils import pretty_print_summary

        pretty_print_summary(summary_dict)


def plot_prob_lnorm(dct, type="values"):
    meanlog, sdlog = dct["meanlog"], dct["sdlog"]
    lb = dct["lb"] if type == "values" else dct["v_lb"]
    ub = dct["ub"] if type == "values" else dct["v_ub"]

    def scale(lb, ub):
        if (ub is None or abs(ub) == np.inf) and (lb is None or abs(lb) == np.inf):
            return 3
        elif ub is not None and abs(ub) != np.inf:
            return max(3, ub)
        elif lb is not None and abs(lb) != np.inf:
            return max(3, lb)

    x_range = np.linspace(0, (np.exp(meanlog) + scale(lb, ub)) * sdlog, 1000)
    y_range = stats.lognorm.pdf(x_range, sdlog, scale=np.exp(meanlog))
    make_colors_continuous(ub, lb, x_range, y_range)
