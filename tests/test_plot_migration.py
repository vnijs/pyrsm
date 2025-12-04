"""Visual comparison tests for plot migration from matplotlib/seaborn to plotnine."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")

# Test data directory
COMPARISON_DIR = Path(__file__).parent / "plot_comparisons"
COMPARISON_DIR.mkdir(exist_ok=True)


class PlotComparison:
    """Helper class to generate side-by-side plot comparisons."""

    @staticmethod
    def save_comparison(
        old_plot_func,
        new_plot_func,
        test_name: str,
        figsize: tuple = (6, 5),
    ):
        """
        Generate and save side-by-side comparison of old vs new plots.

        Parameters
        ----------
        old_plot_func : callable
            Function that generates the old plot (matplotlib/seaborn)
        new_plot_func : callable
            Function that generates the new plot (plotnine)
        test_name : str
            Name for the comparison file
        figsize : tuple
            Figure size for each individual plot
        """
        # Save old plot (matplotlib)
        old_path = COMPARISON_DIR / f"{test_name}_old.png"
        fig, ax = plt.subplots(figsize=figsize)
        old_plot_func(ax)
        fig.tight_layout()
        fig.savefig(old_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save new plot (plotnine)
        new_path = COMPARISON_DIR / f"{test_name}_new.png"
        new_plot = new_plot_func()
        if hasattr(new_plot, "save"):
            new_plot.save(new_path, width=figsize[0], height=figsize[1], dpi=150, verbose=False)

        # Combine into side-by-side comparison image
        comparison_path = COMPARISON_DIR / f"{test_name}_comparison.png"
        try:
            old_img = Image.open(old_path)
            new_img = Image.open(new_path)

            # Create combined image
            total_width = old_img.width + new_img.width + 20  # 20px gap
            max_height = max(old_img.height, new_img.height)
            combined = Image.new("RGB", (total_width, max_height), "white")
            combined.paste(old_img, (0, 0))
            combined.paste(new_img, (old_img.width + 20, 0))
            combined.save(comparison_path)
            print(f"✓ Combined comparison: {comparison_path}")
        except Exception as e:
            print(f"  Warning: Could not create combined image: {e}")

        print(f"  Old: {old_path}")
        print(f"  New: {new_path}")

    @staticmethod
    def generate_test_data():
        """Generate sample data for testing plots."""
        np.random.seed(42)

        # Single variable data
        single_var = pd.DataFrame({"values": np.random.normal(100, 15, 200)})

        # Two group comparison
        two_groups = pd.DataFrame(
            {
                "group": ["A"] * 100 + ["B"] * 100,
                "values": np.concatenate(
                    [np.random.normal(100, 15, 100), np.random.normal(110, 15, 100)]
                ),
            }
        )

        # Proportion data
        prop_data = pd.DataFrame(
            {"category": ["Yes", "No"], "count": [65, 35], "proportion": [0.65, 0.35]}
        )

        # Correlation data
        corr_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 100),
                "x2": np.random.normal(0, 1, 100),
                "x3": np.random.normal(0, 1, 100),
            }
        )
        corr_data["x2"] = corr_data["x1"] * 0.7 + corr_data["x2"] * 0.3
        corr_data["x3"] = corr_data["x1"] * -0.5 + corr_data["x3"] * 0.5

        return {
            "single_var": single_var,
            "two_groups": two_groups,
            "prop_data": prop_data,
            "corr_data": corr_data,
        }


def test_single_mean_migration():
    """Test single_mean plot migration."""
    from pyrsm.basics.single_mean import single_mean

    test_data = PlotComparison.generate_test_data()
    data = {"test_data": test_data["single_var"]}

    # Create single_mean object
    sm = single_mean(data=data, var="values", comp_value=100)

    def old_plot(ax):
        """Generate old matplotlib plot for comparison."""
        # Convert polars to pandas for old-style plotting
        sm.data.to_pandas()[sm.var].plot.hist(ax=ax, color="slateblue", bins=10)
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        ylim = ax.get_ylim()
        ax.vlines(
            x=(sm.comp_value, sm.ci[0], sm.mean, sm.ci[1]),
            ymin=ylim[0],
            ymax=ylim[1],
            colors=("r", "k", "k", "k"),
            linestyles=("solid", "dashed", "solid", "dashed"),
        )

    def new_plot():
        """Generate new plotnine plot."""
        return sm.plot(plots="hist", theme="modern")

    PlotComparison.save_comparison(old_plot, new_plot, "single_mean_hist")


if __name__ == "__main__":
    print("Generating plot comparisons...\n")
    test_single_mean_migration()
    print(f"\n✓ All comparisons saved to: {COMPARISON_DIR}")
    print("Please review the images visually to ensure plots match.")
