from typing import Optional

import numpy as np
import polars as pl
from plotnine import (
    aes,
    geom_density,
    geom_histogram,
    ggplot,
    ggtitle,
    labs,
    theme_bw,
)

import pyrsm.basics.plotting_utils as pu


class central_limit_theorem:
    def __init__(
        self,
        dist: str,
        sample_size: int,
        num_samples: int,
        num_bins: int,
        figsize: Optional[tuple[float, float]] = (10, 10),
        **params: float,
    ) -> None:
        self.dist = dist
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.num_bins = max(1, min(num_bins, 50))
        self.figsize = figsize
        self.params = params

    def simulate(self) -> None:
        if self.dist == "normal":
            self.simulate_normal()
        elif self.dist == "binomial":
            self.simulate_binomial()
        elif self.dist == "uniform":
            self.simulate_uniform()
        elif self.dist == "exponential":
            self.simulate_exponential()
        else:
            print("Invalid distribution")

    def simulate_normal(self) -> None:
        mean = self.params["mean"]
        sd = self.params["sd"]
        samples = [
            np.random.normal(loc=mean, scale=sd, size=self.sample_size)
            for _ in range(self.num_samples)
        ]

        self.plot_distribution(samples)

    def simulate_binomial(self) -> None:
        size = self.params["size"]
        prob = self.params["prob"]

        samples = [
            np.random.binomial(n=size, p=prob, size=self.sample_size)
            for _ in range(self.num_samples)
        ]

        self.plot_distribution(samples)

    def simulate_uniform(self) -> None:
        minimum = self.params["min"]
        maximum = self.params["max"]

        samples = [
            np.random.uniform(low=minimum, high=maximum, size=self.sample_size)
            for _ in range(self.num_samples)
        ]

        self.plot_distribution(samples)

    def simulate_exponential(self) -> None:
        rate = self.params["rate"]

        samples = [
            np.random.exponential(scale=rate, size=self.sample_size)
            for _ in range(self.num_samples)
        ]

        self.plot_distribution(samples)

    def plot_distribution(self, samples: list[np.ndarray]):
        """
        Create plots showing the CLT demonstration.

        Returns a dictionary of ggplot objects:
        - 'sample1': Histogram of first sample
        - 'sample_n': Histogram of last sample
        - 'means_hist': Histogram of sample means
        - 'means_density': Density of sample means
        """
        sample_means = [np.mean(sample) for sample in samples]

        plots = {
            'sample1': self._plot_distribution(
                x=samples[0], title="Histogram of sample #1"
            ),
            'sample_n': self._plot_distribution(
                x=samples[-1], title=f"Histogram of sample #{self.num_samples}"
            ),
            'means_hist': self._plot_distribution(
                x=sample_means, title="Histogram of sample means"
            ),
            'means_density': self._plot_distribution(
                x=sample_means, title="Density of sample means", density_plot=True
            ),
        }
        return plots

    def _plot_distribution(
        self, x: list[np.ndarray], title: str, density_plot: bool = False
    ):
        """Create a single histogram or density plot."""
        data = pl.DataFrame({"value": x})

        if density_plot:
            p = (
                ggplot(data, aes(x="value"))
                + geom_density(fill=pu.PlotConfig.FILL, alpha=0.5)
                + labs(x="Value", y="Density")
                + ggtitle(title)
                + theme_bw()
            )
        else:
            p = (
                ggplot(data, aes(x="value"))
                + geom_histogram(bins=self.num_bins, fill=pu.PlotConfig.FILL, alpha=0.7)
                + labs(x="Value", y="Count")
                + ggtitle(title)
                + theme_bw()
            )
        return p


if __name__ == "__main__":
    clt = central_limit_theorem(
        dist="normal",
        sample_size=1000,
        num_samples=1000,
        num_bins=30,
        figsize=(10, 10),
        mean=0,
        sd=1,
    )
    clt.simulate()

    clt = central_limit_theorem(
        dist="binomial",
        sample_size=1000,
        num_samples=1000,
        num_bins=30,
        figsize=(10, 10),
        size=10,
        prob=0.1,
    )

    clt.simulate()
