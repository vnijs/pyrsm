import numpy as np
from scipy import stats
from .utils import iround, check, plot_discrete


def prob_pois(lamb, lb=None, ub=None, plb=None, pub=None):
    if lamb <= 0:
        raise ValueError("Lambda must be positive")
    if lb is not None and lb < 0:
        lb = 0
    if ub is not None and ub < 0:
        ub = 0
    if lb is None:
        p_lb = None
        p_elb = None
    else:
        p_elb = stats.poisson.pmf(lb, lamb)
        p_lelb = stats.poisson.cdf(lb, lamb)
        if lb > 0:
            p_lb = p_lelb - p_elb
        else:
            p_lb = 0
    if ub is None:
        p_ub = None
        p_eub = None
    else:
        p_eub = stats.poisson.pmf(ub, lamb)
        p_leub = stats.poisson.cdf(ub, lamb)
        if ub > 0:
            p_ub = p_leub - p_eub
        else:
            p_ub = 0
    p_int = None
    if lb is not None and ub is not None:
        p_int = max(np.sum([stats.poisson.pmf(k, lamb) for k in range(lb, ub + 1)]), 0)
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
    if plb is None:
        v_lb = None
    else:
        v_lb = stats.poisson.ppf(plb, lamb)
    if pub is None:
        v_ub = None
    else:
        v_ub = stats.poisson.ppf(pub, lamb)
    check(lb, ub, plb, pub)
    return {
        "p_lb": p_lb,
        "p_ub": p_ub,
        "p_int": p_int,
        "v_lb": v_lb,
        "v_ub": v_ub,
        "lamb": lamb,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
        "p_elb": p_elb,
        "p_eub": p_eub,
    }


def summary_prob_pois(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    add(summary_dict, "Distribution", "Poisson")
    lamb = dct["lamb"]
    add(summary_dict, "Lambda", lamb)
    add(summary_dict, "Mean", lamb)
    add(summary_dict, "Variance", lamb)
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        p_elb, p_eub = dct["p_elb"], dct["p_eub"]
        add(summary_dict, "Lower bound", "" if lb is None else lb)
        add(summary_dict, "Upper bound", "" if ub is None else ub)
        if ub is not None or lb is not None:
            if lb is not None:
                add(summary_dict, f"P(X  = {lb})", p_elb)
                if lb > 0:
                    add(summary_dict, f"P(X  < {lb})", p_lb)
                    add(summary_dict, f"P(X <= {lb})", round(p_lb + p_elb, dec))
                add(summary_dict, f"P(X  > {lb})", round(1 - (p_lb + p_elb), dec))
                add(summary_dict, f"P(X >= {lb})", round(1 - p_lb, dec))
            if ub is not None:
                add(summary_dict, f"P(X  = {ub})", p_eub)
                if ub > 0:
                    add(summary_dict, f"P(X  < {ub})", p_ub)
                    add(summary_dict, f"P(X <= {ub})", round(p_ub + p_eub, dec))
                add(summary_dict, f"P(X  > {ub})", round(1 - (p_ub + p_eub), dec))
                add(summary_dict, f"P(X >= {ub})", round(1 - p_ub, dec))
            if lb is not None and ub is not None:
                add(summary_dict, f"P({lb} <= X <= {ub})", p_int)
                add(summary_dict, f"1 - P({lb} <= X <= {ub})", round(1 - p_int, dec))
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]
        add(summary_dict, "Lower bound", "" if plb is None else f"{plb} ({v_lb})")
        add(summary_dict, "Upper bound", "" if pub is None else f"{pub} ({v_ub})")
        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            if plb is not None and plb >= 0:
                add(summary_dict, f"P(X  < {v_lb})", plb)
                add(summary_dict, f"P(X  > {v_lb})", round(1 - plb, dec))
            if pub is not None and pub <= 1:
                add(summary_dict, f"P(X  < {v_ub})", pub)
                add(summary_dict, f"P(X  > {v_ub})", round(1 - pub, dec))

    if ret:
        return summary_dict
    else:
        from .utils import pretty_print_summary

        pretty_print_summary(summary_dict)


def plot_prob_pois(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]
    lmb = dct["lamb"]
    x_range = list(range(int(stats.poisson.ppf(1 - 0.00001, lmb)) + 1))
    y_range = [stats.poisson.pmf(k, lmb) for k in x_range]
    x_array = np.array(x_range)
    y_array = np.array(y_range)
    mask = y_array > 0.00001
    x_filtered = x_array[mask]
    y_filtered = y_array[mask]
    return plot_discrete(x_filtered, y_filtered, lb, ub, title="Poisson Distribution")
