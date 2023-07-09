import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional


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

    def plot_distribution(self, samples: list[np.ndarray]) -> None:
        sample_means = [np.mean(sample) for sample in samples]

        _, axes = plt.subplots(2, 2, figsize=self.figsize)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        plt.axes(axes[0][0])
        self._plot_distribution(x=samples[0], x_label="Histogram of sample #1").plot()

        plt.axes(axes[0][1])
        self._plot_distribution(
            x=samples[-1], x_label="Histogram of sample #" + str(self.num_samples)
        ).plot()

        plt.axes(axes[1][0])
        self._plot_distribution(
            x=sample_means, x_label="Histogram of sample means"
        ).plot()

        plt.axes(axes[1][1])
        axes[1][1].set_ylabel("y")
        self._plot_distribution(
            x=sample_means, x_label="Density of sample means", density_plot=True
        ).plot()

    def _plot_distribution(
        self, x: list[np.ndarray], x_label: str, density_plot: bool = False
    ) -> matplotlib.axes.Axes:
        stat = "count"
        data = {x_label: x}

        data = pd.DataFrame(data)
        if density_plot:
            return sns.kdeplot(data=data, x=x_label, fill=True)

        return sns.histplot(
            data=data,
            x=x_label,
            stat=stat,
            bins=self.num_bins,
        )


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
