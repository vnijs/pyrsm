from pyrsm.basics.probability_calculator_functions import *

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
    # Probability calculator
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
        elif self.distribution == "F":
            self.dct = prob_fdist(**kwargs)
        elif self.distribution == "lnorm":
            self.dct = prob_lnorm(**kwargs)
        elif self.distribution == "norm":
            self.dct = prob_norm(**kwargs)
        elif self.distribution == "pois":
            self.dct = prob_pois(**kwargs)
        elif self.distribution == "t":
            self.dct = prob_tdist(**kwargs)
        elif self.distribution == "unif":
            self.dct = prob_unif(**kwargs)
        else:
            raise ValueError(f"Distribution must be one of {list(pc_dist.keys())}")

    def summary(self, dec=3):
        type = "probs" if ("plb" in self.args) or ("pub" in self.args) else "values"
        if self.distribution == "binom":
            summary_prob_binom(self.dct, type=type, dec=dec)
        elif self.distribution == "chisq":
            summary_prob_chisq(self.dct, type=type, dec=dec)
        elif self.distribution == "disc":
            summary_prob_disc(self.dct, type=type, dec=dec)
        elif self.distribution == "expo":
            summary_prob_expo(self.dct, type=type, dec=dec)
        elif self.distribution == "F":
            summary_prob_fdist(self.dct, type=type, dec=dec)
        elif self.distribution == "lnorm":
            summary_prob_lnorm(self.dct, type=type, dec=dec)
        elif self.distribution == "norm":
            summary_prob_norm(self.dct, type=type, dec=dec)
        elif self.distribution == "pois":
            summary_prob_pois(self.dct, type=type, dec=dec)
        elif self.distribution == "t":
            summary_prob_tdist(self.dct, type=type, dec=dec)
        elif self.distribution == "unif":
            summary_prob_unif(self.dct, type=type, dec=dec)

    def plot(self):
        type = "probs" if ("plb" in self.args) or ("pub" in self.args) else "values"
        if self.distribution == "binom":
            plot_prob_binom(self.dct, type=type)
        elif self.distribution == "chisq":
            plot_prob_chisq(self.dct, type=type)
        elif self.distribution == "disc":
            plot_prob_disc(self.dct, type=type)
        elif self.distribution == "expo":
            plot_prob_expo(self.dct, type=type)
        elif self.distribution == "F":
            plot_prob_fdist(self.dct, type=type)
        elif self.distribution == "lnorm":
            plot_prob_lnorm(self.dct, type=type)
        elif self.distribution == "norm":
            plot_prob_norm(self.dct, type=type)
        elif self.distribution == "pois":
            plot_prob_pois(self.dct, type=type)
        elif self.distribution == "t":
            plot_prob_tdist(self.dct, type=type)
        elif self.distribution == "unif":
            plot_prob_unif(self.dct, type=type)


if __file__ == "__main__":
    pc = prob_calc("binom", n=10, p=0.1168, lb=4)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, pub=0.3)
    pc.summary()
    pc.plot()
