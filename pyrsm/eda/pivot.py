"""
pivot() - Pivot tables and crosstabs.

Examples:
    import pyrsm_mcp as rsm

    # Frequency table (1 variable)
    rsm.eda.pivot(df, rows='cut')

    # Crosstab (2 variables)
    rsm.eda.pivot(df, rows='cut', cols='color')

    # Aggregation
    rsm.eda.pivot(df, rows='cut', cols='color', values='price', agg='mean')

    # With normalization and totals
    rsm.eda.pivot(df, rows='cut', cols='color', normalize='row', totals=True)
"""

from typing import List, Optional, Union
import polars as pl

# Supported aggregation functions
AGG_FUNCTIONS = {
    "count": lambda col: pl.len(),
    "sum": lambda col: pl.col(col).sum(),
    "mean": lambda col: pl.col(col).mean(),
    "median": lambda col: pl.col(col).median(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "std": lambda col: pl.col(col).std(),
    "var": lambda col: pl.col(col).var(),
}

NORMALIZE_OPTIONS = {"row", "column", "total", "none", None}


def pivot(
    df: Union[pl.DataFrame, pl.LazyFrame],
    rows: str,
    cols: Optional[str] = None,
    values: Optional[str] = None,
    agg: str = "count",
    normalize: Optional[str] = None,
    totals: bool = False,
) -> pl.DataFrame:
    """
    Create pivot tables and crosstabs.

    Args:
        df: Polars DataFrame or LazyFrame
        rows: Row variable (required)
        cols: Column variable for crosstab (optional)
        values: Value column for aggregation (optional, uses count if None)
        agg: Aggregation function. Default: 'count' (or 'mean' if values specified)
             Supported: count, sum, mean, median, min, max, std, var
        normalize: Normalization type: 'row', 'column', 'total', or None
        totals: Whether to include row/column totals

    Returns:
        DataFrame with pivot table

    Examples:
        >>> rsm.eda.pivot(diamonds, rows='cut')  # Frequency table
        >>> rsm.eda.pivot(diamonds, rows='cut', cols='color')  # Crosstab
        >>> rsm.eda.pivot(diamonds, rows='cut', cols='color', values='price', agg='mean')
        >>> rsm.eda.pivot(diamonds, rows='cut', cols='color', normalize='row', totals=True)
    """
    # Convert to LazyFrame for consistency
    if isinstance(df, pl.DataFrame):
        lf = df.lazy()
    else:
        lf = df

    # Default agg based on whether values is specified
    if values and agg == "count":
        agg = "mean"

    # Validate agg
    if agg not in AGG_FUNCTIONS:
        raise ValueError(
            f"Unknown aggregation: {agg}\n"
            f"Supported: {', '.join(AGG_FUNCTIONS.keys())}"
        )

    # Validate normalize
    if normalize not in NORMALIZE_OPTIONS:
        raise ValueError(
            f"Unknown normalize option: {normalize}\n"
            f"Supported: row, column, total, none"
        )

    # Build aggregation expression
    if values:
        agg_expr = AGG_FUNCTIONS[agg](values).alias("value")
    else:
        agg_expr = pl.len().alias("value")

    # Frequency table (single variable)
    if not cols:
        result = lf.group_by(rows).agg(agg_expr)

        # Rename 'value' to more descriptive name
        value_col = f"{agg}_{values}" if values else "count"
        result = result.rename({"value": value_col})

        # Collect for further processing
        pivoted = result.collect()

        # Normalization for frequency table
        if normalize and normalize != "none":
            pivoted = pivoted.with_columns(
                (pl.col(value_col) / pl.col(value_col).sum() * 100).alias(f"{value_col}_pct")
            )

        # Totals for frequency table
        if totals:
            # Cast row column to string to allow "Total" label
            pivoted = pivoted.with_columns(pl.col(rows).cast(pl.Utf8))
            # Cast numeric columns to Float64
            for col in pivoted.columns:
                if col != rows:
                    pivoted = pivoted.with_columns(pl.col(col).cast(pl.Float64))

            # Compute totals
            total_vals = {rows: "Total"}
            for col in pivoted.columns:
                if col != rows:
                    total_vals[col] = float(pivoted[col].sum())

            total_row = pl.DataFrame([total_vals])
            pivoted = pl.concat([pivoted, total_row])

        return pivoted

    # Crosstab (two variables) - need to collect for pivot
    group_cols = [rows, cols]
    grouped = lf.group_by(group_cols).agg(agg_expr)

    # Polars LazyFrame.pivot requires collect first
    df_grouped = grouped.collect()

    # Pivot the data
    pivoted = df_grouped.pivot(
        on=cols,
        index=rows,
        values="value",
        aggregate_function=None,  # Already aggregated
    )

    # Get the column names (excluding index)
    data_cols = [c for c in pivoted.columns if c != rows]

    # Cast row column to string if needed (for totals "Total" label)
    if totals or pivoted[rows].dtype in (pl.Categorical, pl.Enum):
        pivoted = pivoted.with_columns(pl.col(rows).cast(pl.Utf8))

    # Cast numeric columns to Float64 for consistency with totals
    for col in data_cols:
        pivoted = pivoted.with_columns(pl.col(col).cast(pl.Float64))

    # Add totals if requested
    if totals:
        # Row totals
        pivoted = pivoted.with_columns(
            pl.sum_horizontal(data_cols).alias("Total")
        )
        data_cols.append("Total")

        # Column totals (as a new row)
        col_totals = {rows: "Total"}
        for col in data_cols:
            col_totals[col] = float(pivoted[col].sum())

        total_row = pl.DataFrame([col_totals])
        pivoted = pl.concat([pivoted, total_row])

    # Normalize if requested
    if normalize and normalize != "none":
        numeric_cols = [c for c in data_cols if c != "Total"] if totals else data_cols

        if normalize == "row":
            # Each row sums to 100%
            row_sums = pivoted.select(pl.sum_horizontal(numeric_cols)).to_series()
            for col in numeric_cols:
                pivoted = pivoted.with_columns(
                    (pl.col(col) / row_sums * 100).alias(col)
                )
            if totals and "Total" in pivoted.columns:
                pivoted = pivoted.with_columns(pl.lit(100.0).alias("Total"))

        elif normalize == "column":
            # Each column sums to 100%
            for col in numeric_cols:
                col_sum = pivoted[col].sum()
                if col_sum > 0:
                    pivoted = pivoted.with_columns(
                        (pl.col(col) / col_sum * 100).alias(col)
                    )

        elif normalize == "total":
            # All cells sum to 100%
            grand_total = sum(pivoted[col].sum() for col in numeric_cols)
            if grand_total > 0:
                for col in numeric_cols:
                    pivoted = pivoted.with_columns(
                        (pl.col(col) / grand_total * 100).alias(col)
                    )

    return pivoted
