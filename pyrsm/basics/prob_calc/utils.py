from math import ceil, floor


def iround(x, dec):
    return x if not isinstance(x, float) else round(x, dec)


def check(lb, ub, plb, pub):
    if (lb is not None and ub is not None and lb > ub) or (
        plb is not None and pub is not None and plb > pub
    ):
        raise ValueError("Please ensure the lower bound is smaller than the upper bound")


def make_barplot(ub, lb, x_range, y_range):
    import matplotlib.pyplot as plt

    colors = make_colors_discrete(ub, lb, x_range)
    fig, ax = plt.subplots()
    ax.bar(x_range, y_range, color=colors, alpha=0.5)
    if len(x_range) <= 20:
        ax.set_xticks(x_range)
    return ax


def make_colors_discrete(ub, lb, x_range):
    if lb is not None and ub is not None:
        colors = [
            "red" if i < lb else "red" if i > ub else "green" if i == lb or i == ub else "blue"
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
    import matplotlib.pyplot as plt

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
        ax.fill_between(x_range, y_range, where=((x_range > ub) | (x_range < lb)), color="salmon")
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
