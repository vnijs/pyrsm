from .prob_calc.binomial import plot_prob_binom, prob_binom, summary_prob_binom
from .prob_calc.chisq import plot_prob_chisq, prob_chisq, summary_prob_chisq
from .prob_calc.discrete import plot_prob_disc, prob_disc, summary_prob_disc
from .prob_calc.exponential import plot_prob_expo, prob_expo, summary_prob_expo
from .prob_calc.fdist import plot_prob_fdist, prob_fdist, summary_prob_fdist
from .prob_calc.lnorm import plot_prob_lnorm, prob_lnorm, summary_prob_lnorm
from .prob_calc.normal import plot_prob_norm, prob_norm, summary_prob_norm
from .prob_calc.poisson import plot_prob_pois, prob_pois, summary_prob_pois
from .prob_calc.tdist import plot_prob_tdist, prob_tdist, summary_prob_tdist
from .prob_calc.uniform import plot_prob_unif, prob_unif, summary_prob_unif

pc_dist = {
    "Binomial": "binom",
    "Chi-squared": "chisq",
    "Discrete": "disc",
    "Exponential": "expo",
    "F": "fdist",
    "Log normal": "lnorm",
    "Normal": "norm",
    "Poisson": "pois",
    "t": "tdist",
    "Uniform": "unif",
}


class prob_calc:
    def __init__(self, distribution: str, **kwargs) -> None:
        self.distribution = distribution
        self.args = kwargs
        if self.distribution == "binom":
            self.dct = prob_binom(**kwargs)
        elif self.distribution == "chisq":
            self.dct = prob_chisq(**kwargs)
        elif self.distribution == "disc":
            self.dct = prob_disc(**kwargs)
        elif self.distribution == "expo":
            self.dct = prob_expo(**kwargs)
        elif self.distribution == "fdist":
            self.dct = prob_fdist(**kwargs)
        elif self.distribution == "lnorm":
            self.dct = prob_lnorm(**kwargs)
        elif self.distribution == "norm":
            self.dct = prob_norm(**kwargs)
        elif self.distribution == "pois":
            self.dct = prob_pois(**kwargs)
        elif self.distribution == "tdist":
            self.dct = prob_tdist(**kwargs)
        elif self.distribution == "unif":
            self.dct = prob_unif(**kwargs)
        else:
            raise ValueError(f"Distribution must be one of {list(pc_dist.keys())}")

    def summary(self, dec=3, ret=False):
        type = "probs" if ("plb" in self.args) or ("pub" in self.args) else "values"
        if self.distribution == "binom":
            pdict = summary_prob_binom(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "chisq":
            pdict = summary_prob_chisq(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "disc":
            pdict = summary_prob_disc(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "expo":
            pdict = summary_prob_expo(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "fdist":
            pdict = summary_prob_fdist(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "lnorm":
            pdict = summary_prob_lnorm(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "norm":
            pdict = summary_prob_norm(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "pois":
            pdict = summary_prob_pois(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "tdist":
            pdict = summary_prob_tdist(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "unif":
            pdict = summary_prob_unif(self.dct, type=type, dec=dec, ret=True)
        if ret:
            return pdict
        else:
            self.pretty_print_summary(pdict)

    @staticmethod
    def pretty_print_summary(pdict):
        nonprob = {}
        prob = {}
        for k, v in pdict.items():
            if k.startswith("P("):
                prob[k] = v
            else:
                nonprob[k] = v

        print(f"Probability Calculator")
        if nonprob:
            maxlen = max(len(k) for k in nonprob.keys())
            for k, v in nonprob.items():
                if v is not None:
                    pad = " " * (maxlen - len(k))
                    print(f"{k}{pad}: {v}")
        # Probability section: ensure '=' is present and aligned after closing parenthesis
        if prob:
            print()
            # Only format if there are probability keys
            prob_keys = list(prob.keys())
            if prob_keys:
                # Find max length for left side (before value)
                maxlen = max(len(k) for k in prob_keys)
                for k, v in prob.items():
                    if v is not None:
                        # Find position after closing parenthesis
                        close_paren = k.find(")")
                        if close_paren != -1:
                            left = k[: close_paren + 1]
                            right = k[close_paren + 1 :].strip()
                            # Add '=' after closing parenthesis, then right part
                            prob_str = f"{left} = {right}" if right else f"{left} ="
                            pad = " " * (maxlen - len(prob_str))
                            print(f"{prob_str}{pad} {v}")
                        else:
                            pad = " " * (maxlen - len(k))
                            print(f"{k}{pad} {v}")

    def plot(self):
        type = "probs" if ("plb" in self.args) or ("pub" in self.args) else "values"
        if self.distribution == "binom":
            return plot_prob_binom(self.dct, type=type)
        elif self.distribution == "chisq":
            return plot_prob_chisq(self.dct, type=type)
        elif self.distribution == "disc":
            return plot_prob_disc(self.dct, type=type)
        elif self.distribution == "expo":
            return plot_prob_expo(self.dct, type=type)
        elif self.distribution == "fdist":
            return plot_prob_fdist(self.dct, type=type)
        elif self.distribution == "lnorm":
            return plot_prob_lnorm(self.dct, type=type)
        elif self.distribution == "norm":
            return plot_prob_norm(self.dct, type=type)
        elif self.distribution == "pois":
            return plot_prob_pois(self.dct, type=type)
        elif self.distribution == "tdist":
            return plot_prob_tdist(self.dct, type=type)
        elif self.distribution == "unif":
            return plot_prob_unif(self.dct, type=type)


if __name__ == "__main__":
    pc = prob_calc("binom", n=10, p=0.3, lb=1, ub=3)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, lb=4)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, ub=2)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, plb=0.3, pub=0.9)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, pub=0.9)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, plb=0.3)
    pc.summary()
    pc.plot()

    pc = prob_binom(n=10, p=0.3, lb=1, ub=3)
    summary_prob_binom(pc, type="values")
    plot_prob_binom(pc, type="values")
    pb = prob_binom(n=10, p=0.3, lb=1)
    summary_prob_binom(pb, type="values")
    plot_prob_binom(pb, type="values")
    pb = prob_binom(n=10, p=0.3, ub=3)
    summary_prob_binom(pb, type="values")
    plot_prob_binom(pb, type="values")

    pb = prob_binom(n=10, p=0.3, plb=0.1, pub=0.8)
    summary_prob_binom(pb, type="probs")
    plot_prob_binom(pb, type="probs")
    pb = prob_binom(n=10, p=0.3, plb=0.1)
    summary_prob_binom(pb, type="probs")
    plot_prob_binom(pb, type="probs")
    pb = prob_binom(n=10, p=0.3, pub=0.8)
    summary_prob_binom(pb, type="probs")
    plot_prob_binom(pb, type="probs")

    v = list(range(1, 7))
    p = [1 / 6] * 6
    pd = prob_disc(v, p, lb=2, ub=5)
    summary_prob_disc(pd, type="values")
    plot_prob_disc(pd, type="values")
    pd = prob_disc(v, p, lb=2)
    summary_prob_disc(pd, type="values")
    plot_prob_disc(pd, type="values")
    pd = prob_disc(v, p, ub=5)
    summary_prob_disc(pd, type="values")
    plot_prob_disc(pd, type="values")

    v = list(range(1, 7))
    p = [2 / 6, 2 / 6, 1 / 12, 1 / 12, 1 / 12, 1 / 12]
    pd = prob_disc(v, p, plb=0.5, pub=0.8)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")
    pd = prob_disc(v, p, plb=0.5)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")

    v = list(range(1, 7))
    p = [2 / 6, 2 / 6, 1 / 12, 1 / 12, 1 / 12, 1 / 12]
    pd = prob_disc(v, p, plb=0.05, pub=0.95)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")

    pd = prob_disc(v, p, pub=0.95)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")

    pd = prob_disc(v, p, plb=0.05)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")

    pf = prob_fdist(df1=10, df2=10, lb=0.5, ub=2.978)
    summary_prob_fdist(pf, type="values")
    plot_prob_fdist(pf, type="values")
    pf = prob_fdist(df1=10, df2=10, lb=0.5)
    summary_prob_fdist(pf, type="values")
    plot_prob_fdist(pf, type="values")
    pf = prob_fdist(df1=10, df2=10, ub=2.978)
    summary_prob_fdist(pf, type="values")
    plot_prob_fdist(pf, type="values")

    pf = prob_fdist(df1=10, df2=10, plb=0.05, pub=0.95)
    summary_prob_fdist(pf, type="probs")
    plot_prob_fdist(pf, type="probs")
    pf = prob_fdist(df1=10, df2=10, plb=0.05)
    summary_prob_fdist(pf, type="probs")
    pf = prob_fdist(df1=10, df2=10, pub=0.95)
    summary_prob_fdist(pf, type="probs")
    plot_prob_fdist(pf, type="probs")

    pn = prob_norm(mean=0, stdev=1, lb=-0.5, ub=0.5)
    summary_prob_norm(pn, type="values")
    plot_prob_norm(pn, type="values")
    pn = prob_norm(mean=0, stdev=1, lb=-0.5)
    summary_prob_norm(pn, type="values")
    plot_prob_norm(pn, type="values")
    pn = prob_norm(mean=0, stdev=1, ub=0.5)
    summary_prob_norm(pn, type="values")
    plot_prob_norm(pn, type="values")

    pn = prob_norm(mean=0, stdev=1, plb=0.025, pub=0.975)
    summary_prob_norm(pn, type="probs")
    plot_prob_norm(pn, type="probs")
    pn = prob_norm(mean=0, stdev=1, plb=0.025)
    summary_prob_norm(pn, type="probs")
    plot_prob_norm(pn, type="probs")
    pn = prob_norm(mean=0, stdev=1, pub=0.975)
    summary_prob_norm(pn, type="probs")
    plot_prob_norm(pn, type="probs")

    pc = prob_chisq(df=1, lb=1, ub=3.841)
    summary_prob_chisq(pc, type="values")
    plot_prob_chisq(pc, type="values")
    pc = prob_chisq(df=1, lb=1)
    summary_prob_chisq(pc, type="values")
    plot_prob_chisq(pc, type="values")
    pc = prob_chisq(df=1, ub=3.841)
    summary_prob_chisq(pc, type="values")
    plot_prob_chisq(pc, type="values")

    pc = prob_chisq(df=1, plb=0.05, pub=0.95)
    summary_prob_chisq(pc, type="probs")
    plot_prob_chisq(pc, type="probs")
    pc = prob_chisq(df=1, plb=0.05)
    summary_prob_chisq(pc, type="probs")
    plot_prob_chisq(pc, type="probs")
    pc = prob_chisq(df=1, pub=0.95)
    summary_prob_chisq(pc, type="probs")
    plot_prob_chisq(pc, type="probs")

    pu = prob_unif(min=0, max=1, lb=0.2, ub=0.8)
    summary_prob_unif(pu, type="values")
    plot_prob_unif(pu, type="values")
    pu = prob_unif(min=0, max=1, lb=0.2)
    summary_prob_unif(pu, type="values")
    plot_prob_unif(pu, type="values")
    pu = prob_unif(min=0, max=1, ub=0.8)
    summary_prob_unif(pu, type="values")
    plot_prob_unif(pu, type="values")

    pu = prob_unif(min=0, max=1, plb=0.2, pub=0.8)
    summary_prob_unif(pu, type="probs")
    plot_prob_unif(pu, type="probs")
    pu = prob_unif(min=0, max=1, plb=0.2)
    summary_prob_unif(pu, type="probs")
    plot_prob_unif(pu, type="probs")
    pu = prob_unif(min=0, max=1, pub=0.8)
    summary_prob_unif(pu, type="probs")
    plot_prob_unif(pu, type="probs")

    pt = prob_tdist(df=10, lb=-2.228, ub=2.228)
    summary_prob_tdist(pt, type="values")
    plot_prob_tdist(pt, type="values")
    pt = prob_tdist(df=10, ub=2.228)
    summary_prob_tdist(pt, type="values")
    plot_prob_tdist(pt, type="values")
    pt = prob_tdist(df=10, lb=-2.228)
    plot_prob_tdist(pt, type="values")
    pt = prob_tdist(df=10)
    summary_prob_tdist(pt, type="values")
    plot_prob_tdist(pt, type="values")

    pt = prob_tdist(df=10, plb=0.025, pub=0.975)
    summary_prob_tdist(pt, type="probs")
    plot_prob_tdist(pt, type="prob")
    pt = prob_tdist(df=10, pub=0.975)
    summary_prob_tdist(pt, type="probs")
    plot_prob_tdist(pt, type="probs")
    pt = prob_tdist(df=10, plb=0.025)
    summary_prob_tdist(pt, type="probs")
    plot_prob_tdist(pt, type="probs")
