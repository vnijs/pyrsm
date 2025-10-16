import numpy as np
from .utils import iround, check, make_colors_discrete


def prob_disc(v, p, lb=None, ub=None, plb=None, pub=None):
    if len(v) != len(p):
        if len(v) % len(p) == 0:
            p *= len(v) // len(p)
        else:
            raise ValueError(
                "The number of values must be the same or a multiple of the number of probabilities"
            )
    if not 0.999 <= sum(p) <= 1.001:
        raise ValueError("Probabilities for a discrete variable do not sum to 1")
    vp_sorted = sorted(zip(v, p))
    v, p = zip(*vp_sorted)

    def ddisc(b):
        return p[v.index(b)] if b in v else 0

    def pdisc(b):
        return sum(p[i] for i in range(len(v)) if v[i] < b)

    def qdisc(prob):
        return next((v[i] for i in range(len(p)) if sum(p[: i + 1]) >= prob), None)

    result = {"v": v, "p": p}
    if lb is not None:
        result.update(
            {"lb": lb, "p_elb": ddisc(lb), "p_lb": pdisc(lb), "p_lelb": ddisc(lb) + pdisc(lb)}
        )
    if ub is not None:
        result.update(
            {"ub": ub, "p_eub": ddisc(ub), "p_ub": pdisc(ub), "p_leub": ddisc(ub) + pdisc(ub)}
        )
    if lb is not None and ub is not None:
        result["p_int"] = result["p_leub"] - result["p_lb"]
    if plb is not None:
        vlb = qdisc(plb)
        if vlb is not None:
            result.update(
                {
                    "plb": plb,
                    "vlb": vlb,
                    "vp_elb": ddisc(vlb),
                    "vp_lb": pdisc(vlb),
                    "vp_lelb": ddisc(vlb) + pdisc(vlb),
                }
            )
    if pub is not None:
        vub = qdisc(pub)
        if vub is not None:
            result.update(
                {
                    "pub": pub,
                    "vub": vub,
                    "vp_eub": ddisc(vub),
                    "vp_ub": pdisc(vub),
                    "vp_leub": ddisc(vub) + pdisc(vub),
                }
            )
    if (
        plb is not None
        and pub is not None
        and result.get("vlb") is not None
        and result.get("vub") is not None
    ):
        result["vp_int"] = result["vp_leub"] - result["vp_lb"]
    check(lb, ub, plb, pub)
    return result


def summary_prob_disc(dct, type="values", dec=3, ret=False):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    lb, ub = dct.get("lb", None), dct.get("ub", None)
    v, p = dct["v"], dct["p"]
    mean = sum(vi * pi for vi, pi in zip(v, p))
    std_dev = (sum(pi * (vi - mean) ** 2 for vi, pi in zip(v, p))) ** 0.5
    header = {
        "Distribution": "Discrete",
        "Values": v,
        "Probabilities": [round(i, dec) for i in p],
        "Mean": round(mean, dec),
        "St. dev": round(std_dev, dec),
        "Lower bound": None,
        "Upper bound": None,
    }
    summary = {}
    if type == "values":
        if lb is not None:
            header.update({"Lower bound": lb})
            summary.update({f"P(X = {lb})": round(dct.get("p_elb", 0), dec)})
            if lb > min(v):
                summary.update(
                    {f"P(X < {lb})": dct.get("p_lb", 0), f"P(X <= {lb})": dct.get("p_lelb", 0)}
                )
            if lb < max(v):
                summary.update(
                    {
                        f"P(X > {lb})": round(1 - (dct.get("p_lb", 0) + dct.get("p_elb", 0)), dec),
                        f"P(X >= {lb})": round(1 - dct.get("p_lb", 0), dec),
                    }
                )
        if ub is not None:
            header.update({"Upper bound": ub})
            summary.update({f"P(X = {ub})": dct.get("p_eub", 0)})
            if ub > min(v):
                summary.update(
                    {f"P(X < {ub})": dct.get("p_ub", 0), f"P(X <= {ub})": dct.get("p_leub", 0)}
                )
            if ub < max(v):
                summary.update(
                    {
                        f"P(X > {ub})": round(1 - (dct.get("p_ub", 0) + dct.get("p_eub", 0)), dec),
                        f"P(X >= {ub})": round(1 - dct.get("p_ub", 0), dec),
                    }
                )
        if lb is not None and ub is not None:
            summary.update(
                {
                    f"P({lb} <= X <= {ub})": dct.get("p_int", 0),
                    f"1 - P({lb} <= X <= {ub})": round(1 - dct.get("p_int", 0), dec),
                }
            )
    else:
        plb, pub, vlb, vub = (
            dct.get("plb", None),
            dct.get("pub", None),
            dct.get("vlb", None),
            dct.get("vub", None),
        )
        if plb is not None:
            header.update({"Lower bound": f"{plb} ({dct.get('vlb')})"})
            summary.update({f"P(X = {vlb})": dct.get("vp_elb", 0)})
            if vlb > min(v):
                summary.update(
                    {f"P(X < {vlb})": dct.get("vp_lb", 0), f"P(X <= {vlb})": dct.get("vp_lelb", 0)}
                )
            if vlb < max(v):
                summary.update(
                    {
                        f"P(X > {vlb})": round(
                            1 - (dct.get("vp_lb", 0) + dct.get("vp_elb", 0)), dec
                        ),
                        f"P(X >= {vlb})": round(1 - dct.get("vp_lb", 0), dec),
                    }
                )
        if pub is not None:
            header.update({"Upper bound": f"{pub} ({dct.get('vub')})"})
            summary.update({f"P(X = {vub})": dct.get("vp_eub", 0)})
            if vub > min(v):
                summary.update(
                    {f"P(X < {vub})": dct.get("vp_ub", 0), f"P(X <= {vub})": dct.get("vp_leub", 0)}
                )
            if vub < max(v):
                summary.update(
                    {
                        f"P(X > {vub})": round(
                            1 - (dct.get("vp_ub", 0) + dct.get("vp_eub", 0)), dec
                        ),
                        f"P(X >= {vub})": round(1 - dct.get("vp_ub", 0), dec),
                    }
                )
        if plb is not None and pub is not None:
            summary.update(
                {
                    f"P({vlb} <= X <= {vub})": dct.get("vp_int", 0),
                    f"1 - P({vlb} <= X <= {vub})": round(1 - dct.get("vp_int", 0), dec),
                }
            )
    if ret:
        result = {**header, **summary}
        return result
    else:
        for k, v in header.items():
            if v is not None:
                print(f"{k}: {v}")
        for k, v in summary.items():
            print(f"{k}: {v}")


def plot_prob_disc(dct, type="values"):
    if type == "values":
        lb, ub = dct.get("lb", None), dct.get("ub", None)
    else:
        lb, ub = dct.get("vlb", None), dct.get("vub", None)
    x_range, y_range = dct["v"], dct["p"]
    import matplotlib.pyplot as plt

    colors = make_colors_discrete(ub, lb, x_range)
    fig, ax = plt.subplots()
    ax.bar(range(len(x_range)), y_range, tick_label=x_range, color=colors, alpha=0.5)
