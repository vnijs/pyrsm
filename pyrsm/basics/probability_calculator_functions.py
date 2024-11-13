import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import numpy as np
from scipy import stats
from math import floor, ceil


def iround(x, dec):
    return x if not isinstance(x, float) else round(x, dec)


def check(lb, ub, plb, pub):
    if (lb is not None and ub is not None and lb > ub) or (
        plb is not None and pub is not None and plb > pub
    ):
        raise ValueError(
            "Please ensure the lower bound is smaller than the upper bound"
        )


def make_colors_discrete(ub, lb, x_range):
    if lb is not None and ub is not None:
        colors = [
            "red"
            if i < lb
            else "red"
            if i > ub
            else "green"
            if i == lb or i == ub
            else "blue"
            for i in x_range
        ]
    elif lb is not None:
        colors = ["red" if i < lb else "green" if i == lb else "blue" for i in x_range]
    elif ub is not None:
        colors = ["blue" if i < ub else "green" if i == ub else "red" for i in x_range]
    else:
        colors = "blue"
    return colors


def make_colors_continuous(ub, lb, x_range, y_range):
    fig, ax = plt.subplots()
    ax.plot(x_range, y_range, "k")
    if ub is not None and lb is not None:
        ax.fill_between(
            x_range,
            y_range,
            where=((x_range > lb) & (x_range < ub)),
            color="blue",
            alpha=0.5,
        )
        ax.fill_between(
            x_range, y_range, where=((x_range > ub) | (x_range < lb)), color="salmon"
        )
    elif ub is not None:
        ax.fill_between(
            x_range,
            y_range,
            where=(x_range < ub),
            color="blue",
            alpha=0.5,
        )
        ax.fill_between(x_range, y_range, where=(x_range > ub), color="salmon")
    elif lb is not None:
        ax.fill_between(
            x_range,
            y_range,
            where=(x_range > lb),
            color="blue",
            alpha=0.5,
        )
        ax.fill_between(x_range, y_range, where=(x_range < lb), color="salmon")
    else:
        ax.fill_between(
            x_range,
            y_range,
            where=((x_range > 0) | (x_range < 0)),
            color="blue",
            alpha=0.5,
        )

    if lb is not None:
        ax.axvline(lb, color="black", linestyle="dashed", linewidth=0.5)
    if ub is not None:
        ax.axvline(ub, color="black", linestyle="dashed", linewidth=0.5)

    return ax


def make_barplot(ub, lb, x_range, y_range):
    fig, ax = plt.subplots()
    colors = make_colors_discrete(ub, lb, x_range)
    ax.bar(x_range, y_range, color=colors, alpha=0.5)
    if len(x_range) <= 20:
        ax.set_xticks(x_range)

    return ax


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


def summary_prob_binom(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    print("Probability calculator")
    print("Distribution: Binomial")

    n, p = dct["n"], dct["p"]
    print("n           :", n)
    print("p           :", p)
    print("Mean        :", round(n * p, dec))
    print("St. dev     :", round(np.sqrt(n * p * (1 - p)), dec))

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        p_elb, p_eub = dct["p_elb"], dct["p_eub"]
        p_lelb, p_leub = dct["p_lelb"], dct["p_leub"]

        print("Lower bound :", "" if lb is None else lb)
        print("Upper bound :", "" if ub is None else ub, "\n")

        if ub is not None or lb is not None:
            if lb is not None:
                # print(f"P(X  = {lb}) = {p_elb} (green)")
                print(f"P(X  = {lb}) = {p_elb}")
                if lb > 0:
                    print(f"P(X  < {lb}) = {p_lb}")
                    print(f"P(X <= {lb}) = {p_lelb}")
                if lb < n:
                    print(f"P(X  > {lb}) = {round(1 - (p_lb + p_elb), dec)}")
                    print(f"P(X >= {lb}) = {round(1 - p_lb, dec)}")

            if ub is not None:
                print(f"P(X  = {ub}) = {p_eub}")
                if ub > 0:
                    print(f"P(X  < {ub}) = {p_ub}")
                    print(f"P(X <= {ub}) = {p_leub}")
                if ub < n:
                    print(f"P(X  > {ub}) = {round(1 - (p_ub + p_eub), dec)}")
                    print(f"P(X >= {ub}) = {round(1 - p_ub, dec)}")

            if lb is not None and ub is not None:
                print(f"P({lb} <= X <= {ub})     = {p_int}")
                print(f"1 - P({lb} <= X <= {ub}) = {round(1 - p_int, dec)}")
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["vlb"], dct["vub"]
        vp_lb, vp_ub, vp_int = dct["vp_lb"], dct["vp_ub"], dct["vp_int"]
        vp_elb, vp_eub = dct["vp_elb"], dct["vp_eub"]
        vp_lelb, vp_leub = dct["vp_lelb"], dct["vp_leub"]

        print("Lower bound :", "" if plb is None else f"{plb} ({v_lb})")
        print("Upper bound :", "" if pub is None else f"{pub} ({v_ub})\n")

        if pub is not None or plb is not None:
            if plb is not None:
                print(f"P(X  = {v_lb}) = {vp_elb}")
                if v_lb > 0:
                    print(f"P(X  < {v_lb}) = {vp_lb}")
                    print(f"P(X <= {v_lb}) = {vp_lelb}")
                if v_lb < n:
                    print(f"P(X  > {v_lb}) = {round(1 - (vp_lb + vp_elb), dec)}")
                    print(f"P(X >= {v_lb}) = {round(1 - vp_lb, dec)}")

            if pub is not None:
                print(f"P(X  = {v_ub}) = {vp_eub}")
                if v_ub > 0:
                    print(f"P(X  < {v_ub}) = {vp_ub}")
                    print(f"P(X <= {v_ub}) = {vp_leub}")
                if v_ub < n:
                    print(f"P(X  > {v_ub}) = {round(1 - (vp_ub + vp_eub), dec)}")
                    print(f"P(X >= {v_ub}) = {round(1 - vp_ub, dec)}")

            if plb is not None and pub is not None:
                print(f"P({v_lb} <= X <= {v_ub})     = {vp_int}")
                print(f"1 - P({v_lb} <= X <= {v_ub}) = {round(1 - vp_int, dec)}")


def plot_prob_binom(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["vlb"], dct["vub"]

    n, p = dct["n"], dct["p"]

    x_range = np.arange(0, n)
    df = (
        pd.DataFrame()
        .assign(x_range=x_range, y_range=stats.binom.pmf(x_range, n, p))
        .query("y_range > 0.00001")
    )

    make_barplot(ub, lb, df.x_range, df.y_range)


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


def summary_prob_chisq(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    print("Probability calculator")
    print("Distribution: Chi-square")

    df = dct["df"]
    print("Df          :", df)
    print("Mean        :", df)
    print("Variance    :", 2 * df)

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]

        print("Lower bound :", "0" if lb is None else lb)
        print("Upper bound :", "Inf" if ub is None else ub)

        if ub is not None or lb is not None:
            print()

            if lb is not None:
                print(f"P(X < {lb}) = {p_lb}")
                print(f"P(X > {lb}) = {round(1 - p_lb, dec)}")

            if ub is not None:
                print(f"P(X < {ub}) = {p_ub}")
                print(f"P(X > {ub}) = {round(1 - p_ub, dec)}")

            if lb is not None and ub is not None:
                print(f"P({lb} < X < {ub})     = {p_int}")
                print(f"1 - P({lb} < X < {ub}) = {round(1 - p_int, dec)}")
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]

        print("Lower bound :", "0" if plb is None or plb < 0 else plb)
        print("Upper bound :", "1" if pub is None or pub > 1 else pub)

        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            print()

            if plb is not None and plb >= 0:
                print(f"P(X < {v_lb}) = {plb}")
                print(f"P(X > {v_lb}) = {round(1 - plb, dec)}")

            if pub is not None and pub <= 1:
                print(f"P(X < {v_ub}) = {pub}")
                print(f"P(X > {v_ub}) = {round(1 - pub, dec)}")

            if (plb is not None and plb >= 0) and (pub is not None and pub <= 1):
                print(f"P({v_lb} < X < {v_ub})     = {round(pub - plb, dec)}")
                print(f"1 - P({v_lb} < X < {v_ub}) = {round(1 - (pub - plb), dec)}")


def plot_prob_chisq(dct, type="values"):
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


def prob_disc(v, p, lb=None, ub=None, plb=None, pub=None):
    # Check lengths
    if len(v) != len(p):
        if len(v) % len(p) == 0:
            p *= len(v) // len(p)
        else:
            raise ValueError(
                "The number of values must be the same or a multiple of the number of probabilities"
            )

    # Ensure sum of probabilities is approximately 1
    if not 0.999 <= sum(p) <= 1.001:
        raise ValueError("Probabilities for a discrete variable do not sum to 1")

    # Sort values and probabilities in ascending order of values
    vp_sorted = sorted(zip(v, p))
    v, p = zip(*vp_sorted)

    # Discrete distribution functions
    def ddisc(b):
        return p[v.index(b)] if b in v else 0

    def pdisc(b):
        return sum(p[i] for i in range(len(v)) if v[i] < b)

    def qdisc(prob):
        return next((v[i] for i in range(len(p)) if sum(p[: i + 1]) >= prob), None)

    # Calculate probabilities
    result = {"v": v, "p": p}

    if lb is not None:
        result.update(
            {
                "lb": lb,
                "p_elb": ddisc(lb),
                "p_lb": pdisc(lb),
                "p_lelb": ddisc(lb) + pdisc(lb),
            }
        )
    if ub is not None:
        result.update(
            {
                "ub": ub,
                "p_eub": ddisc(ub),
                "p_ub": pdisc(ub),
                "p_leub": ddisc(ub) + pdisc(ub),
            }
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


def summary_prob_disc(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    lb, ub = dct.get("lb", None), dct.get("ub", None)
    v, p = dct["v"], dct["p"]
    mean = sum(vi * pi for vi, pi in zip(v, p))
    std_dev = (sum(pi * (vi - mean) ** 2 for vi, pi in zip(v, p))) ** 0.5

    header = {
        "Distribution :": "Discrete",
        "Values       :": v,
        "Probabilities:": [round(i, dec) for i in p],
        "Mean         :": round(mean, dec),
        "St. dev      :": round(std_dev, dec),
        "Lower bound  :": None,
        "Upper bound  :": None,
    }
    summary = {}

    if type == "values":
        if lb is not None:
            header.update({"Lower bound  :": lb})
            summary.update({f"P(X  = {lb}) =": round(dct.get("p_elb", 0), dec)})
            if lb > min(v):
                summary.update(
                    {
                        f"P(X  < {lb}) =": dct.get("p_lb", 0),
                        f"P(X <= {lb}) =": dct.get("p_lelb", 0),
                    }
                )
            if lb < max(v):
                summary.update(
                    {
                        f"P(X  > {lb}) =": round(
                            1 - (dct.get("p_lb", 0) + dct.get("p_elb", 0)), dec
                        ),
                        f"P(X >= {lb}) =": round(1 - dct.get("p_lb", 0), dec),
                    }
                )

        if ub is not None:
            header.update({"Upper bound  :": ub})
            summary.update({f"P(X  = {ub}) =": dct.get("p_eub", 0)})
            if ub > min(v):
                summary.update(
                    {
                        f"P(X  < {ub}) =": dct.get("p_ub", 0),
                        f"P(X <= {ub}) =": dct.get("p_leub", 0),
                    }
                )
            if ub < max(v):
                summary.update(
                    {
                        f"P(X  > {ub}) =": round(
                            1 - (dct.get("p_ub", 0) + dct.get("p_eub", 0)), dec
                        ),
                        f"P(X >= {ub}) =": round(1 - dct.get("p_ub", 0), dec),
                    }
                )

        if lb is not None and ub is not None:
            summary.update(
                {
                    f"P({lb} <= X <= {ub})     =": dct.get("p_int", 0),
                    f"1 - P({lb} <= X <= {ub}) =": round(1 - dct.get("p_int", 0), dec),
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
            header.update({"Lower bound  :": f"{plb} ({dct.get('vlb')})"}),
            summary.update({f"P(X  = {vlb}) =": dct.get("vp_elb", 0)})
            if vlb > min(v):
                summary.update(
                    {
                        f"P(X  < {vlb}) = ": dct.get("vp_lb", 0),
                        f"P(X <= {vlb}) =": dct.get("vp_lelb", 0),
                    }
                )
            if vlb < max(v):
                summary.update(
                    {
                        f"P(X  > {vlb}) =": round(
                            1 - (dct.get("vp_lb", 0) + dct.get("vp_elb", 0)), dec
                        ),
                        f"P(X >= {vlb}) =": round(1 - dct.get("vp_lb", 0), dec),
                    }
                )

        if pub is not None:
            header.update({"Upper bound  :": f"{pub} ({dct.get('vub')})"})
            summary.update({f"P(X  = {vub}) =": dct.get("vp_eub", 0)})
            if vub > min(v):
                summary.update(
                    {
                        f"P(X  < {vub}) =": dct.get("vp_ub", 0),
                        f"P(X <= {vub}) =": dct.get("vp_leub", 0),
                    }
                )
            if vub < max(v):
                summary.update(
                    {
                        f"P(X  > {vub}) = ": round(
                            1 - (dct.get("vp_ub", 0) + dct.get("vp_eub", 0)), dec
                        ),
                        f"P(X >= {vub}) =": round(1 - dct.get("vp_ub", 0), dec),
                    }
                )

        if plb is not None and pub is not None:
            summary.update(
                {
                    f"P({vlb} <= X <= {vub})     =": dct.get("vp_int", 0),
                    f"1 - P({vlb} <= X <= {vub}) =": round(
                        1 - dct.get("vp_int", 0), dec
                    ),
                }
            )

    for k, v in header.items():
        if v is not None:
            print(f"{k} {v}")

    for k, v in summary.items():
        print(f"{k} {v}")


def plot_prob_disc(dct, type="values"):
    if type == "values":
        lb, ub = dct.get("lb", None), dct.get("ub", None)
    else:
        lb, ub = dct.get("vlb", None), dct.get("vub", None)
    x_range, y_range = dct["v"], dct["p"]

    fig, ax = plt.subplots()
    colors = make_colors_discrete(ub, lb, x_range)
    ax.bar(range(len(x_range)), y_range, tick_label=x_range, color=colors, alpha=0.5)


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
            raise ValueError(
                "Please ensure the lower bound is smaller than the upper bound value"
            )
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


def summary_prob_expo(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    print("Probability calculator")
    print("Distribution: Exponential")

    rate = dct["rate"]
    print("Rate        :", rate)
    print("Mean        :", round(1 / rate, dec))
    print("Variance    :", round(rate**-2, dec))

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]

        print("Lower bound :", "0" if lb is None else lb)
        print("Upper bound :", "Inf" if ub is None else ub)

        if ub is not None or lb is not None:
            print()

            if lb is not None:
                print(f"P(X < {lb}) = {p_lb}")
                print(f"P(X > {lb}) = {round(1 - p_lb, dec)}")

            if ub is not None:
                print(f"P(X < {ub}) = {p_ub}")
                print(f"P(X > {ub}) = {round(1 - p_ub, dec)}")

            if lb is not None and ub is not None:
                print(f"P({lb} < X < {ub})     = {p_int}")
                print(f"1 - P({lb} < X < {ub}) = {round(1 - p_int, dec)}")
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]

        print("Lower bound :", "0" if plb is None or plb < 0 else plb)
        print("Upper bound :", "1" if pub is None or pub > 1 else pub)

        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            print()

            if plb is not None and plb >= 0:
                print(f"P(X < {v_lb}) = {plb}")
                print(f"P(X > {v_lb}) = {round(1 - plb, dec)}")

            if pub is not None and pub <= 1:
                print(f"P(X < {v_ub}) = {pub}")
                print(f"P(X > {v_ub}) = {round(1 - pub, dec)}")

            if (plb is not None and plb >= 0) and (pub is not None and pub <= 1):
                print(f"P({v_lb} < X < {v_ub})     = {pub - plb}")
                print(f"1 - P({v_lb} < X < {v_ub}) = {round(1 - (pub - plb), dec)}")


def plot_prob_expo(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]

    rate = dct["rate"]
    x_range = np.linspace(0, stats.expon.ppf(0.99, scale=1 / rate), 1000)
    y_range = stats.expon.pdf(x_range, scale=1 / rate)
    make_colors_continuous(ub, lb, x_range, y_range)


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


def summary_prob_fdist(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    print("Probability calculator")
    print("Distribution: F")

    df1, df2 = dct["df1"], dct["df2"]
    print("Df 1        :", df1)
    print("Df 2        :", df2)
    m = round(df2 / (df2 - 2), dec) if df2 > 2 else "NA"
    variance = (
        round(
            (2 * df2**2 * (df1 + df2 - 2)) / (df1 * (df2 - 2) ** 2 * (df2 - 4)), dec
        )
        if df2 > 4
        else np.nan
    )
    print("Mean        :", m)
    print("Variance    :", variance)

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]

        print("Lower bound :", "0" if lb is None else lb)
        print("Upper bound :", "Inf" if ub is None else ub)

        if ub is not None or lb is not None:
            print()

            if lb is not None:
                lb = round(lb, dec)
                print(f"P(X < {lb}) = {p_lb}")
                print(f"P(X > {lb}) = {round(1 - p_lb, dec)}")

            if ub is not None:
                ub = round(ub, dec)
                print(f"P(X < {ub}) = {p_ub}")
                print(f"P(X > {ub}) = {round(1 - p_ub, dec)}")

            if lb is not None and ub is not None:
                print(f"P({lb} < X < {ub})     = {p_int}")
                print(f"1 - P({lb} < X < {ub}) = {round(1 - p_int, dec)}")
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]

        print("Lower bound :", "0" if plb is None or plb < 0 else plb)
        print("Upper bound :", "1" if pub is None or pub > 1 else pub)

        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            print()

            if plb is not None and plb >= 0:
                v_lb = round(v_lb, dec)
                print(f"P(X < {v_lb}) = {plb}")
                print(f"P(X > {v_lb}) = {round(1 - plb, dec)}")

            if pub is not None and pub <= 1:
                v_ub = round(v_ub, dec)
                print(f"P(X < {v_ub}) = {pub}")
                print(f"P(X > {v_ub}) = {round(1 - pub, dec)}")

            if (plb is not None and plb >= 0) and (pub is not None and pub <= 1):
                print(f"P({v_lb} < X < {v_ub})     = {round(pub - plb, dec)}")
                print(f"1 - P({v_lb} < X < {v_ub}) = {round(1 - (pub - plb), dec)}")


def plot_prob_fdist(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]

    df1, df2 = dct["df1"], dct["df2"]
    x_range = np.linspace(0, stats.f.ppf(0.99, df1, df2), 1000)
    y_range = stats.f.pdf(x_range, df1, df2)
    make_colors_continuous(ub, lb, x_range, y_range)


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

    v_ub = (
        stats.lognorm.ppf(pub, sdlog, scale=np.exp(meanlog))
        if pub is not None
        else None
    )
    v_lb = (
        stats.lognorm.ppf(plb, sdlog, scale=np.exp(meanlog))
        if plb is not None
        else None
    )

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


def summary_prob_lnorm(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    meanlog, sdlog = dct["meanlog"], dct["sdlog"]
    lb, v_lb = dct["lb"], dct["v_lb"]
    ub, v_ub = dct["ub"], dct["v_ub"]
    p_lb, plb = dct["p_lb"], dct["plb"]
    p_ub, pub = dct["p_ub"], dct["pub"]
    p_int = dct["p_int"]
    print(
        f"Probability calculator\nDistribution: Log Normal\nMean log    : {round(meanlog, dec)}"
    )
    print(f"St. dev log : {round(sdlog, dec)}")

    if type == "values":
        print(f"Lower bound : {lb if lb is not None else '-Inf'}")
        print(f"Upper bound : {ub if ub is not None else 'Inf'}\n")
        if ub != np.inf and lb != -np.inf:
            print(f"P(X < {lb if lb is not None else '-Inf'}) = {p_lb}")
            print(f"P(X > {lb if lb is not None else '-Inf'}) = {round(1 - p_lb, dec)}")
            print(f"P(X < {ub if ub is not None else 'Inf'}) = {p_ub}")
            print(f"P(X > {ub if ub is not None else 'Inf'}) = {round(1 - p_ub, dec)}")
            print(
                f"P({lb if lb is not None else '-Inf'} < X < {ub if ub is not None else 'Inf'})     = {round(p_int, dec)}"
            )
            print(
                f"1 - P({lb if lb is not None else '-Inf'} < X < {ub if ub is not None else 'Inf'}) = {round(1 - p_int, dec)}"
            )
        elif lb != -np.inf:
            print(f"P(X < {lb if lb is not None else '-Inf'}) = {p_lb}")
            print(f"P(X > {lb if lb is not None else '-Inf'}) = {round(1 - p_lb, dec)}")
        elif ub != np.inf:
            print(f"P(X < {ub if ub is not None else 'Inf'}) = {p_ub}")
            print(f"P(X > {ub if ub is not None else 'Inf'}) = {round(1 - p_ub, dec)}")
    else:
        if pub is None:
            pub = 2
        if plb is None:
            plb = -1

        print("Lower bound :", "0" if plb < 0 else plb, "(" + str(v_lb) + ")")
        print("Upper bound :", "1" if pub > 1 else pub, "(" + str(v_ub) + ")")

        if pub <= 1 or plb >= 0:
            print()

            if plb >= 0:
                print(f"P(X < {v_lb}) = {plb}")
                print(f"P(X > {v_lb}) = {round(1 - plb, dec)}")

            if pub <= 1:
                print(f"P(X < {v_ub}) = {pub}")
                print(f"P(X > {v_ub}) = {round(1 - pub, dec)}")

            if pub <= 1 and plb >= 0:
                print(f"P({v_lb} < X < {v_ub})     = {round(pub - plb, dec)}")
                print(f"1 - P({v_lb} < X < {v_ub}) = {round(1 - (pub - plb), dec)}")


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


def prob_norm(mean, stdev, lb=None, ub=None, plb=None, pub=None):
    if lb is not None:
        p_lb = stats.norm.cdf(lb, mean, stdev)
    else:
        p_lb = None

    if ub is not None:
        p_ub = stats.norm.cdf(ub, mean, stdev)
    else:
        p_ub = None

    if lb is not None and ub is not None:
        p_int = max(p_ub - p_lb, 0)
    else:
        p_int = None

    if plb is not None:
        v_lb = stats.norm.ppf(plb, mean, stdev)
    else:
        v_lb = None

    if pub is not None:
        v_ub = stats.norm.ppf(pub, mean, stdev)
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
        "stdev": stdev,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
    }


def summary_prob_norm(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    print("Probability calculator")
    print("Distribution: Normal")

    mean, stdev = dct["mean"], dct["stdev"]
    print("Mean        :", round(mean, dec))
    print("St. dev     :", round(stdev, dec))

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]

        print("Lower bound :", "-Inf" if lb is None else lb)
        print("Upper bound :", "Inf" if ub is None else ub)

        if ub is not None or lb is not None:
            print()

            if lb is not None:
                lb = round(lb, dec)
                print(f"P(X < {lb}) = {p_lb}")
                print(f"P(X > {lb}) = {round(1 - p_lb, dec)}")

            if ub is not None:
                ub = round(ub, dec)
                print(f"P(X < {ub}) = {p_ub}")
                print(f"P(X > {ub}) = {round(1 - p_ub, dec)}")

            if lb is not None and ub is not None:
                print(f"P({lb} < X < {ub})     = {p_int}")
                print(f"1 - P({lb} < X < {ub}) = {round(1 - p_int, dec)}")
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]

        print("Lower bound :", "0" if plb is None or plb < 0 else plb)
        print("Upper bound :", "1" if pub is None or pub > 1 else pub)

        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            print()

            if plb is not None and plb >= 0:
                v_lb = round(v_lb, dec)
                print(f"P(X < {v_lb}) = {plb}")
                print(f"P(X > {v_lb}) = {round(1 - plb, dec)}")

            if pub is not None and pub <= 1:
                v_ub = round(v_ub, dec)
                print(f"P(X < {v_ub}) = {pub}")
                print(f"P(X > {v_ub}) = {round(1 - pub, dec)}")

            if (plb is not None and plb >= 0) and (pub is not None and pub <= 1):
                print(f"P({v_lb} < X < {v_ub})     = {round(pub - plb, dec)}")
                print(f"1 - P({v_lb} < X < {v_ub}) = {round(1 - (pub - plb), dec)}")


def plot_prob_norm(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]

    mean, stdev = dct["mean"], dct["stdev"]
    x_range = np.linspace(mean - 3 * stdev, mean + 3 * stdev, 1000)
    y_range = stats.norm.pdf(x_range, mean, stdev)
    make_colors_continuous(ub, lb, x_range, y_range)


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


def summary_prob_pois(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    print("Probability calculator")
    print("Distribution: Poisson")

    lamb = dct["lamb"]
    print("Lambda      :", lamb)
    print("Mean        :", lamb)
    print("Variance    :", lamb)

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        p_elb, p_eub = dct["p_elb"], dct["p_eub"]

        print("Lower bound :", "" if lb is None else lb)
        print("Upper bound :", "" if ub is None else ub)

        if ub is not None or lb is not None:
            print()

            if lb is not None:
                print(f"P(X  = {lb}) = {p_elb}")
                if lb > 0:
                    print(f"P(X  < {lb}) = {p_lb}")
                    print(f"P(X <= {lb}) = {round(p_lb + p_elb, dec)}")
                print(f"P(X  > {lb}) = {round(1 - (p_lb + p_elb), dec)}")
                print(f"P(X >= {lb}) = {round(1 - p_lb, dec)}")

            if ub is not None:
                print(f"P(X  = {ub}) = {p_eub}")
                if ub > 0:
                    print(f"P(X  < {ub}) = {p_ub}")
                    print(f"P(X <= {ub}) = {round(p_ub + p_eub, dec)}")
                print(f"P(X  > {ub}) = {round(1 - (p_ub + p_eub), dec)}")
                print(f"P(X >= {ub}) = {round(1 - p_ub, dec)}")

            if lb is not None and ub is not None:
                print(f"P({lb} <= X <= {ub})     = {p_int}")
                print(f"1 - P({lb} <= X <= {ub}) = {round(1 - p_int, dec)}")
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]

        print("Lower bound :", "" if plb is None else f"{plb} ({v_lb})")
        print("Upper bound :", "" if pub is None else f"{pub} ({v_ub})")

        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            print()

            if plb is not None and plb >= 0:
                print(f"P(X  < {v_lb}) = {plb}")
                print(f"P(X  > {v_lb}) = {round(1 - plb, dec)}")

            if pub is not None and pub <= 1:
                print(f"P(X  < {v_ub}) = {pub}")
                print(f"P(X  > {v_ub}) = {round(1 - pub, dec)}")


def plot_prob_pois(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]

    lmb = dct["lamb"]
    x_range = range(int(stats.poisson.ppf(1 - 0.00001, lmb)))
    y_range = [stats.poisson.pmf(k, lmb) for k in x_range]

    df = (
        pd.DataFrame()
        .assign(x_range=x_range, y_range=y_range)
        .query("y_range > 0.00001")
    )

    make_barplot(ub, lb, df.x_range, df.y_range)


def prob_tdist(df, lb=None, ub=None, plb=None, pub=None):
    if lb is None:
        p_lb = None
    else:
        p_lb = stats.t.cdf(lb, df)

    if ub is None:
        p_ub = None
    else:
        p_ub = stats.t.cdf(ub, df)

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
        v_lb = stats.t.ppf(plb, df)
    else:
        v_lb = None

    if pub is not None:
        v_ub = stats.t.ppf(pub, df)
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


def summary_prob_tdist(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    print("Probability calculator")
    print("Distribution: t")

    df = dct["df"]
    n = df + 1
    print("Df          :", df)
    print("Mean        :", 0)
    if n > 2:
        print("St. dev     :", round(n / (n - 2), dec))
    else:
        print("St. dev     :", "NA")

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]

        print("Lower bound :", "-Inf" if lb is None else lb)
        print("Upper bound :", "Inf" if ub is None else ub)

        if ub is not None or lb is not None:
            print()

            if lb is not None:
                print(f"P(X < {lb}) = {p_lb}")
                print(f"P(X > {lb}) = {round(1 - p_lb, dec)}")

            if ub is not None:
                ub = round(ub, dec)
                print(f"P(X < {ub}) = {p_ub}")
                print(f"P(X > {ub}) = {round(1 - p_ub, dec)}")

            if lb is not None and ub is not None:
                print(f"P({lb} < X < {ub})     = {p_int}")
                print(f"1 - P({lb} < X < {ub}) = {round(1 - p_int, dec)}")
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]

        print("Lower bound :", "0" if plb is None or plb < 0 else plb)
        print("Upper bound :", "1" if pub is None or pub > 1 else pub)

        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            print()

            if plb is not None and plb >= 0:
                v_lb = round(v_lb, dec)
                print(f"P(X < {v_lb}) = {plb}")
                print(f"P(X > {v_lb}) = {round(1 - plb, dec)}")

            if pub is not None and pub <= 1:
                v_ub = round(v_ub, dec)
                print(f"P(X < {v_ub}) = {pub}")
                print(f"P(X > {v_ub}) = {round(1 - pub, dec)}")

            if (plb is not None and plb >= 0) and (pub is not None and pub <= 1):
                print(f"P({v_lb} < X < {v_ub})     = {round(pub - plb, dec)}")
                print(f"1 - P({v_lb} < X < {v_ub}) = {round(1 - (pub - plb), dec)}")


def plot_prob_tdist(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]

    x_range = np.linspace(-3, 3, 1000)
    y_range = stats.t.pdf(x_range, dct["df"])
    make_colors_continuous(ub, lb, x_range, y_range)


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


def summary_prob_unif(dct, type="values", dec=3):
    dct = {k: iround(v, dec) for k, v in dct.items()}
    print("Probability calculator")
    print("Distribution: Uniform")

    min, max = dct["min"], dct["max"]
    mean, stdev = dct["mean"], dct["stdev"]

    print("Min         :", min)
    print("Max         :", max)
    print("Mean        :", round(mean, dec))
    print("St. dev     :", round(stdev, dec))

    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]

        print("Lower bound :", min if lb is None else lb)
        print("Upper bound :", max if ub is None else ub)

        if ub is not None or lb is not None:
            print()

            if lb is not None:
                print(f"P(X < {lb}) = {p_lb}")
                print(f"P(X > {lb}) = {round(1 - p_lb, dec)}")

            if ub is not None:
                print(f"P(X < {ub}) = {p_ub}")
                print(f"P(X > {ub}) = {round(1 - p_ub, dec)}")

            if lb is not None and ub is not None:
                print(f"P({lb} < X < {ub})     = {p_int}")
                print(f"1 - P({lb} < X < {ub}) = {round(1 - p_int, dec)}")
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]

        print("Lower bound :", "0" if plb is None or plb < 0 else plb)
        print("Upper bound :", "1" if pub is None or pub > 1 else pub)

        if (pub is None or pub <= 1) or (plb is None or plb >= 0):
            print()

            if plb is not None and plb >= 0:
                print(f"P(X < {v_lb}) = {plb}")
                print(f"P(X > {v_lb}) = {round(1 - plb, dec)}")

            if pub is not None and pub <= 1:
                print(f"P(X < {v_ub}) = {pub}")
                print(f"P(X > {v_ub}) = {round(1 - pub, dec)}")

            if (plb is not None and plb >= 0) and (pub is not None and pub <= 1):
                print(f"P({v_lb} < X < {v_ub})     = {round(pub - plb, dec)}")
                print(f"1 - P({v_lb} < X < {v_ub}) = {round(1 - (pub - plb), dec)}")


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
