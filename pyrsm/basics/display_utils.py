"""Shared display utilities for basics module output formatting."""

import polars as pl
from typing import Callable


def is_notebook() -> bool:
    """Detect if running in a Jupyter/IPython notebook environment."""
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None and "IPKernelApp" in ipy.config:
            return True
    except (ImportError, AttributeError):
        pass
    return False


def format_pval(p: float, dec: int = 3) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "< .001"
    return str(round(p, dec))


def print_plain_tables(*tables: pl.DataFrame, max_rows: int = -1) -> None:
    """Print polars DataFrames with consistent plain-text formatting."""
    with pl.Config(
        tbl_rows=max_rows,
        tbl_cols=-1,
        fmt_str_lengths=100,
        tbl_width_chars=200,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        tbl_hide_dtype_separator=True,
    ):
        for i, table in enumerate(tables):
            print(table)
            if i < len(tables) - 1:
                print()


def print_sig_codes() -> None:
    """Print significance codes footer."""
    print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


class SummaryDisplay:
    """
    Unified summary display handler for basics modules.

    Provides dual-mode output: styled tables in notebooks via great_tables,
    plain text tables in terminals via polars Config.

    Parameters
    ----------
    header_func : callable
        Function that prints the summary header (metadata block).
    plain_func : callable
        Function that prints plain text tables. Signature: (extra, dec) -> None
    styled_func : callable, optional
        Function that displays styled tables. Signature: (extra, dec) -> None
        If None, plain_func is used in all environments.
    """

    def __init__(
        self,
        header_func: Callable[[], None],
        plain_func: Callable[[bool, int], None],
        styled_func: Callable[[bool, int], None] | None = None,
    ):
        self.header_func = header_func
        self.plain_func = plain_func
        self.styled_func = styled_func

    def display(self, extra: bool = False, dec: int = 3) -> None:
        """Display summary with environment-appropriate formatting."""
        self.header_func()

        if is_notebook() and self.styled_func is not None:
            self.styled_func(extra, dec)
        else:
            self.plain_func(extra, dec)

        print_sig_codes()


def style_table(
    df: pl.DataFrame,
    title: str = "",
    subtitle: str = "",
    number_cols: list[str] | None = None,
    integer_cols: list[str] | None = None,
    dec: int = 3,
):
    """
    Apply great_tables styling to a polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to style.
    title : str
        Table title.
    subtitle : str
        Table subtitle.
    number_cols : list[str], optional
        Columns to format as decimal numbers.
    integer_cols : list[str], optional
        Columns to format as integers.
    dec : int
        Decimal places for number formatting.

    Returns
    -------
    GT object for display in notebooks.
    """
    gt = df.style.tab_header(title=title, subtitle=subtitle)

    if number_cols:
        existing = [c for c in number_cols if c in df.columns]
        if existing:
            gt = gt.fmt_number(columns=existing, decimals=dec, use_seps=False)

    if integer_cols:
        existing = [c for c in integer_cols if c in df.columns]
        if existing:
            gt = gt.fmt_integer(columns=existing, use_seps=False)

    return gt.tab_options(table_margin_left="0px")


def display_styled(*tables):
    """Display styled tables in a notebook."""
    from IPython.display import display
    for table in tables:
        display(table)
