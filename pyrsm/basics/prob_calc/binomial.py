import numpy as np
from scipy import stats

from .utils import check, iround, make_barplot


def prob_binom(n: int, p: float, lb=None, ub=None, plb=None, pub=None):
    # Helper function to convert to int if not None
    def to_int(x):
        return int(max(0, x)) if x is not None else None

    lb = to_int(lb)
    ub = to_int(ub)
    (
        p_elb,
        p_lelb,
        p_lb,
        p_eub,
        p_leub,
        p_ub,
        p_int,
        vlb,
        vp_elb,
        vp_lelb,
        vp_lb,
        vub,
        vp_eub,
        vp_leub,
        vp_ub,
        vp_int,
    ) = [None] * 16

    if lb is not None:
        if lb > n:
            lb = n
        p_elb = stats.binom.pmf(lb, n, p)
        p_lelb = stats.binom.cdf(lb, n, p)
        if lb > 0:
            p_lb = sum(stats.binom.pmf(range(0, lb), n, p))
        else:
            p_lb = 0

    if ub is not None:
        if ub > n:
            ub = n
        p_eub = stats.binom.pmf(ub, n, p)
        p_leub = stats.binom.cdf(ub, n, p)
        if ub > 0:
            p_ub = sum(stats.binom.pmf(range(0, ub), n, p))
        else:
            p_ub = 0

    if lb is not None and ub is not None:
        p_int = sum(stats.binom.pmf(range(lb, ub + 1), n, p))

    if plb is not None:
        if plb > 1:
            plb = 1
        if plb < 0:
            plb = 0
        vlb = stats.binom.ppf(plb, n, p)
        vp_elb = stats.binom.pmf(vlb, n, p)
        vp_lelb = stats.binom.cdf(vlb, n, p)
        if vlb > 0:
            vp_lb = sum(stats.binom.pmf(range(0, int(vlb)), n, p))
        else:
            vp_lb = 0

    if pub is not None:
        if pub > 1:
            pub = 1
        if pub < 0:
            pub = 0
        vub = stats.binom.ppf(pub, n, p)
        vp_eub = stats.binom.pmf(vub, n, p)
        vp_leub = stats.binom.cdf(vub, n, p)
        if vub > 0:
            vp_ub = sum(stats.binom.pmf(range(0, int(vub)), n, p))
        else:
            vp_ub = 0

    check(lb, ub, plb, pub)

    if plb is not None and pub is not None:
        vp_int = sum(stats.binom.pmf(range(int(vlb), int(vub) + 1), n, p))

    # Return a dictionary of results
    return {
        "n": n,
        "p": p,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
        "p_elb": p_elb,
        "p_lb": p_lb,
        "p_eub": p_eub,
        "p_ub": p_ub,
        "p_int": p_int,
        "p_lelb": p_lelb,
        "p_leub": p_leub,
        "vlb": vlb,
        "vub": vub,
        "vp_elb": vp_elb,
        "vp_lb": vp_lb,
        "vp_eub": vp_eub,
        "vp_ub": vp_ub,
        "vp_lelb": vp_lelb,
        "vp_leub": vp_leub,
        "vp_int": vp_int,
    }


def summary_prob_binom(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    add(summary_dict, "Probability calculator", None)
    add(summary_dict, "Distribution", "Binomial")
    n, p = dct["n"], dct["p"]
    add(summary_dict, "n", n)
    add(summary_dict, "p", p)
    add(summary_dict, "Mean", round(n * p, dec))
    add(summary_dict, "St. dev", round(np.sqrt(n * p * (1 - p)), dec))
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        p_elb, p_eub = dct["p_elb"], dct["p_eub"]
        p_lelb, p_leub = dct["p_lelb"], dct["p_leub"]
        add(summary_dict, "Lower bound", lb if lb is not None else "")
        add(summary_dict, "Upper bound", ub if ub is not None else "")
        if ub is not None or lb is not None:
            if lb is not None:
                add(summary_dict, f"P(X  = {lb})", p_elb)
                if lb > 0:
                    add(summary_dict, f"P(X  < {lb})", p_lb)
                    add(summary_dict, f"P(X <= {lb})", p_lelb)
                if lb < n:
                    add(summary_dict, f"P(X  > {lb})", round(1 - (p_lb + p_elb), dec))
                    add(summary_dict, f"P(X >= {lb})", round(1 - p_lb, dec))
            if ub is not None:
                add(summary_dict, f"P(X  = {ub})", p_eub)
                if ub > 0:
                    add(summary_dict, f"P(X  < {ub})", p_ub)
                    add(summary_dict, f"P(X <= {ub})", p_leub)
                if ub < n:
                    add(summary_dict, f"P(X  > {ub})", round(1 - (p_ub + p_eub), dec))
                    add(summary_dict, f"P(X >= {ub})", round(1 - p_ub, dec))
            if lb is not None and ub is not None:
                add(summary_dict, f"P({lb} <= X <= {ub})", p_int)
                add(summary_dict, f"1 - P({lb} <= X <= {ub})", round(1 - p_int, dec))
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["vlb"], dct["vub"]
        vp_lb, vp_ub, vp_int = dct["vp_lb"], dct["vp_ub"], dct["vp_int"]
        vp_elb, vp_eub = dct["vp_elb"], dct["vp_eub"]
        vp_lelb, vp_leub = dct["vp_lelb"], dct["vp_leub"]
        add(summary_dict, "Lower bound", f"{plb} ({v_lb})" if plb is not None else "")
        add(summary_dict, "Upper bound", f"{pub} ({v_ub})" if pub is not None else "")
        if pub is not None or plb is not None:
            if plb is not None:
                add(summary_dict, f"P(X  = {v_lb})", vp_elb)
                if v_lb > 0:
                    add(summary_dict, f"P(X  < {v_lb})", vp_lb)
                    add(summary_dict, f"P(X <= {v_lb})", vp_lelb)
                if v_lb < n:
                    add(summary_dict, f"P(X  > {v_lb})", round(1 - (vp_lb + vp_elb), dec))
                    add(summary_dict, f"P(X >= {v_lb})", round(1 - vp_lb, dec))
            if pub is not None:
                add(summary_dict, f"P(X  = {v_ub})", vp_eub)
                if v_ub > 0:
                    add(summary_dict, f"P(X  < {v_ub})", vp_ub)
                    add(summary_dict, f"P(X <= {v_ub})", vp_leub)
                if v_ub < n:
                    add(summary_dict, f"P(X  > {v_ub})", round(1 - (vp_ub + vp_eub), dec))
                    add(summary_dict, f"P(X >= {v_ub})", round(1 - vp_ub, dec))
            if plb is not None and pub is not None:
                add(summary_dict, f"P({v_lb} <= X <= {v_ub})", vp_int)
                add(summary_dict, f"1 - P({v_lb} <= X <= {v_ub})", round(1 - vp_int, dec))
    if ret:
        return summary_dict
    else:
        for k, v in summary_dict.items():
            if v is not None:
                print(f"{k}: {v}")


def plot_prob_binom(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["vlb"], dct["vub"]
    n, p = dct["n"], dct["p"]
    x_range = np.arange(0, n)
    y_range = stats.binom.pmf(x_range, n, p)
    mask = y_range > 0.00001
    x_filtered = x_range[mask]
    y_filtered = y_range[mask]
    make_barplot(ub, lb, x_filtered, y_filtered)
