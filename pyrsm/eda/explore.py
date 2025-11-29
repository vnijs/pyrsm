"""
explore() - Summary statistics for numeric columns.

Examples:
    import pyrsm_mcp as rsm

    # All numeric columns, default stats
    rsm.eda.explore(df)

    # Specific columns
    rsm.eda.explore(df, cols=['price', 'carat'])

    # Custom functions
    rsm.eda.explore(df, cols=['price'], funs=['mean', 'median', 'std'])

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
    "var": lambda col: pl.col(col).var(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "count": lambda col: pl.col(col).count(),
    "n_unique": lambda col: pl.col(col).n_unique(),
    "null_count": lambda col: pl.col(col).null_count(),
}

DEFAULT_FUNS = ["mean", "std", "min", "max", "count"]

# Numeric dtypes for auto-detection
NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def explore(
    df: Union[pl.DataFrame, pl.LazyFrame],
    cols: Optional[List[str]] = None,
    funs: Optional[List[str]] = None,
    by: Optional[str] = None,
) -> pl.DataFrame:
    """
    Compute summary statistics for numeric columns.

    Args:
        df: Polars DataFrame or LazyFrame
        cols: Column names to summarize. If None, uses all numeric columns.
        funs: Functions to compute. Default: ['mean', 'std', 'min', 'max', 'count']
              Supported: mean, median, sum, std, var, min, max, count, n_unique, null_count
        by: Optional column to group by

    Returns:
        DataFrame with summary statistics

    Examples:
        >>> rsm.eda.explore(diamonds)  # All numeric, default stats
        >>> rsm.eda.explore(diamonds, cols=['price', 'carat'])
        >>> rsm.eda.explore(diamonds, cols=['price'], funs=['mean', 'median'])
        >>> rsm.eda.explore(diamonds, cols=['price'], by='cut')
    """
    # Convert to LazyFrame for consistency
    if isinstance(df, pl.DataFrame):
        lf = df.lazy()
    else:
        lf = df

    # Default functions
    if funs is None:
        funs = DEFAULT_FUNS

    # Validate functions
    for fun in funs:
        if fun not in EXPLORE_FUNCTIONS:
            raise ValueError(
                f"Unknown function: {fun}\n"
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
        for fun in funs:
            expr = EXPLORE_FUNCTIONS[fun](col).alias(f"{col}_{fun}")
            exprs.append(expr)

    # Execute with or without grouping
    if by:
        result = lf.group_by(by).agg(exprs)
    else:
        result = lf.select(exprs)

    return result.collect()
