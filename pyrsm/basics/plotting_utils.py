"""Shared plotting utilities for pyrsm.basics classes."""

from typing import Literal

from plotnine import (
    aes,
    geom_hline,
    geom_vline,
    theme,
    theme_bw,
    theme_classic,
    theme_gray,
    theme_minimal,
    element_text,
    element_rect,
    element_blank,
)


class PlotConfig:
    """Default configuration for pyrsm plots."""

    # Primary fill colors
    FILL = "slateblue"

    # Reference line colors
    COMP_COLOR = "red"  # Null hypothesis / comparison value
    STAT_COLOR = "black"  # Sample statistics (mean, CI bounds)

    @staticmethod
    def theme():
        """Default pyrsm theme with white background and grid lines."""
        return theme_bw()


class PlotTheme:
    """Manage consistent theming across pyrsm plots."""

    @staticmethod
    def get_theme(
        theme_name: Literal["modern", "publication", "minimal", "classic"] = "modern",
    ):
        """
        Get a predefined plotnine theme.

        Parameters
        ----------
        theme_name : str
            Theme name: 'modern', 'publication', 'minimal', or 'classic'

        Returns
        -------
        theme
            A plotnine theme object
        """
        themes = {
            "modern": theme_minimal()
            + theme(
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
                axis_line=element_rect(color="black", size=0.5),
                plot_title=element_text(size=12, weight="bold"),
            ),
            "publication": theme_bw()
            + theme(
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
                plot_title=element_text(size=11, weight="bold"),
                axis_text=element_text(size=9),
            ),
            "minimal": theme_minimal(),
            "classic": theme_classic(),
        }
        return themes.get(theme_name, themes["modern"])


class ReferenceLine:
    """Helper class for adding reference lines to plots."""

    @staticmethod
    def vline(x, color="black", linetype="solid", alpha=1.0):
        """Add vertical reference line."""
        return geom_vline(xintercept=x, color=color, linetype=linetype, alpha=alpha)

    @staticmethod
    def hline(y, color="black", linetype="solid", alpha=1.0):
        """Add horizontal reference line."""
        return geom_hline(yintercept=y, color=color, linetype=linetype, alpha=alpha)

    @staticmethod
    def ci_vlines(mean, ci_lower, ci_upper, comp_value=None):
        """
        Add confidence interval vertical lines for single mean tests.

        Parameters
        ----------
        mean : float
            Sample mean
        ci_lower : float
            Lower confidence bound
        ci_upper : float
            Upper confidence bound
        comp_value : float, optional
            Comparison/null hypothesis value

        Returns
        -------
        list
            List of geom_vline objects
        """
        lines = [
            ReferenceLine.vline(mean, color="black", linetype="solid"),
            ReferenceLine.vline(ci_lower, color="black", linetype="dashed"),
            ReferenceLine.vline(ci_upper, color="black", linetype="dashed"),
        ]
        if comp_value is not None:
            lines.insert(0, ReferenceLine.vline(comp_value, color="red", linetype="solid"))
        return lines

    @staticmethod
    def ci_hlines(mean, ci_lower, ci_upper, comp_value=None):
        """
        Add confidence interval horizontal lines for comparison tests.

        Parameters
        ----------
        mean : float
            Sample mean
        ci_lower : float
            Lower confidence bound
        ci_upper : float
            Upper confidence bound
        comp_value : float, optional
            Comparison/null hypothesis value

        Returns
        -------
        list
            List of geom_hline objects
        """
        lines = [
            ReferenceLine.hline(mean, color="black", linetype="solid"),
            ReferenceLine.hline(ci_lower, color="black", linetype="dashed"),
            ReferenceLine.hline(ci_upper, color="black", linetype="dashed"),
        ]
        if comp_value is not None:
            lines.insert(0, ReferenceLine.hline(comp_value, color="red", linetype="solid"))
        return lines

    @staticmethod
    def significance_levels():
        """Add standard significance level reference lines."""
        return [
            ReferenceLine.hline(1.96, color="blue", linetype="dotted", alpha=0.5),
            ReferenceLine.hline(-1.96, color="blue", linetype="dotted", alpha=0.5),
            ReferenceLine.hline(1.64, color="blue", linetype="dotted", alpha=0.3),
            ReferenceLine.hline(-1.64, color="blue", linetype="dotted", alpha=0.3),
            ReferenceLine.hline(0, color="blue", linetype="solid", alpha=0.7),
        ]


class PlotExporter:
    """Utilities for saving plots consistently."""

    @staticmethod
    def save(
        plot,
        filename: str,
        width: float = 8,
        height: float = 6,
        dpi: int = 300,
        format: str = "png",
    ):
        """
        Save a plotnine plot with consistent settings.

        Parameters
        ----------
        plot : plotnine.ggplot
            The plot object to save
        filename : str
            Output filename (with or without extension)
        width : float
            Width in inches
        height : float
            Height in inches
        dpi : int
            Resolution in dots per inch
        format : str
            File format (png, pdf, svg)
        """
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        plot.save(filename, width=width, height=height, dpi=dpi, verbose=False)


def ci_label(alt: str = "two-sided", conf: float = 0.95, dec: int = 3) -> tuple[str, str]:
    """
    Generate confidence interval labels for hypothesis tests.

    This function is imported from pyrsm.basics.utils for backward compatibility.

    Parameters
    ----------
    alt : str
        Alternative hypothesis type ('two-sided', 'greater', 'less')
    conf : float
        Confidence level
    dec : int
        Decimal places for rounding

    Returns
    -------
    tuple[str, str]
        Lower and upper CI label strings
    """
    import pyrsm.basics.utils as bu

    return bu.ci_label(alt, conf, dec)
