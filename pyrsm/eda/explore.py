"""
explore() - Summary statistics for numeric columns.

Examples:
    import pyrsm as rsm

    # All numeric columns, default stats
    rsm.eda.explore(df)

    # Specific columns
    rsm.eda.explore(df, cols=['price', 'carat'])

    # Custom aggregation functions
    rsm.eda.explore(df, cols=['price'], agg=['mean', 'median', 'sd'])

    # Grouped
    rsm.eda.explore(df, cols=['price'], by='cut')
"""

from typing import List, Optional, Union
import polars as pl

# Supported summary functions
EXPLORE_FUNCTIONS = {
    "mean": lambda col: pl.col(col).mean(),
    "median": lambda col: pl.col(col).median(),
    "sum": lambda col: pl.col(col).sum(),
    "std": lambda col: pl.col(col).std(),
    "sd": lambda col: pl.col(col).std(),  # Alias for std (R-style)
    "var": lambda col: pl.col(col).var(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "count": lambda col: pl.col(col).count(),
    "n": lambda col: pl.col(col).count(),  # Alias for count
    "n_unique": lambda col: pl.col(col).n_unique(),
    "n_missing": lambda col: pl.col(col).null_count(),  # Alias for null_count
    "null_count": lambda col: pl.col(col).null_count(),
}

DEFAULT_AGG = ["mean", "median", "min", "max", "sd"]

# Numeric dtypes for auto-detection
NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def explore(
    df: Union[pl.DataFrame, pl.LazyFrame],
    cols: Optional[List[str]] = None,
    agg: Optional[List[str]] = None,
    by: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute summary statistics for numeric columns.

    Args:
        df: Polars DataFrame or LazyFrame
        cols: Column names to summarize. If None, uses all numeric columns.
        agg: Aggregation functions to compute. Default: ['mean', 'median', 'min', 'max', 'sd']
             Supported: mean, median, sum, std, sd, var, min, max, count, n, n_unique, n_missing, null_count
        by: Optional column to group by

    Returns:
        DataFrame with summary statistics

    Examples:
        >>> rsm.eda.explore(diamonds)  # All numeric, default agg
        >>> rsm.eda.explore(diamonds, cols=['price', 'carat'])
        >>> rsm.eda.explore(diamonds, cols=['price'], agg=['mean', 'median'])
        >>> rsm.eda.explore(diamonds, cols=['price'], by='cut')
    """
    # Convert to LazyFrame for consistency
    if isinstance(df, pl.DataFrame):
        lf = df.lazy()
    else:
        lf = df

    # Default aggregation functions
    if agg is None:
        agg = DEFAULT_AGG

    # Validate aggregation functions
    for func in agg:
        if func not in EXPLORE_FUNCTIONS:
            raise ValueError(
                f"Unknown aggregation function: {func}\n"
                f"Supported: {', '.join(EXPLORE_FUNCTIONS.keys())}"
            )

    # Auto-detect numeric columns if none specified
    if cols is None:
        schema = lf.collect_schema()
        cols = [
            name for name, dtype in schema.items()
            if isinstance(dtype, NUMERIC_DTYPES) or dtype in NUMERIC_DTYPES
        ]
        if not cols:
            raise ValueError("No numeric columns found in dataset")

    # Build aggregation expressions
    exprs = []
    for col in cols:
        for func in agg:
            expr = EXPLORE_FUNCTIONS[func](col).alias(f"{col}_{func}")
            exprs.append(expr)

    # Execute with or without grouping
    if by:
        result = lf.group_by(by).agg(exprs).collect()
    else:
        # Without grouping: transpose to have stats as rows, variables as columns
        wide_result = lf.select(exprs).collect()

        # Build transposed table: rows = agg functions, columns = variables
        rows = []
        for func in agg:
            row = {"statistic": func}
            for col in cols:
                row[col] = wide_result[f"{col}_{func}"][0]
            rows.append(row)
        result = pl.DataFrame(rows)

    return result
