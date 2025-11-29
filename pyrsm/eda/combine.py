"""
combine() - Combine datasets using joins, binds, or set operations.

Adapted from R's radiant.data::combine_data().

Examples:
    import pyrsm as rsm

    # Join datasets
    rsm.eda.combine(orders, customers, by='customer_id', type='inner_join')
    rsm.eda.combine(orders, customers, by='cust_id:customer_id', type='left_join')

    # Bind operations
    rsm.eda.combine(df1, df2, type='bind_rows')
    rsm.eda.combine(df1, df2, type='bind_cols')

    # Set operations
    rsm.eda.combine(df1, df2, type='intersect')
"""

from typing import List, Optional, Union
import polars as pl

# Combine types matching Radiant
JOIN_TYPES = {"inner_join", "left_join", "right_join", "full_join", "semi_join", "anti_join"}
BIND_TYPES = {"bind_rows", "bind_cols"}
SET_TYPES = {"intersect", "union", "setdiff"}
ALL_TYPES = JOIN_TYPES | BIND_TYPES | SET_TYPES

# Map Radiant names to Polars how= parameter
POLARS_JOIN_MAP = {
    "inner_join": "inner",
    "left_join": "left",
    "right_join": "right",
    "full_join": "full",
    "semi_join": "semi",
    "anti_join": "anti",
}


def _parse_by(by: str) -> tuple:
    """Parse by= string into (left_on, right_on) lists.

    Examples:
        "customer_id" -> (["customer_id"], ["customer_id"])
        "cust_id:customer_id" -> (["cust_id"], ["customer_id"])
        "a,b" -> (["a", "b"], ["a", "b"])
        "a:x,b:y" -> (["a", "b"], ["x", "y"])
    """
    left_on, right_on = [], []
    for key in by.split(","):
        key = key.strip()
        if ":" in key:
            left, right = key.split(":", 1)
            left_on.append(left.strip())
            right_on.append(right.strip())
        else:
            left_on.append(key)
            right_on.append(key)
    return left_on, right_on


def combine(
    x: Union[pl.DataFrame, pl.LazyFrame],
    y: Union[pl.DataFrame, pl.LazyFrame],
    by: Optional[str] = None,
    type: str = "inner_join",
    add: Optional[List[str]] = None,
    suffix: str = "_right",
) -> pl.DataFrame:
    """
    Combine two datasets using joins, binds, or set operations.

    Args:
        x: Left/first dataset (Polars DataFrame or LazyFrame)
        y: Right/second dataset to combine with x
        by: Join keys. Format: "col" or "col1,col2" or "left_col:right_col"
            Required for join operations, ignored for bind/set operations.
        type: Combine type. Default: 'inner_join'
              Joins (require by=): inner_join, left_join, right_join, full_join, semi_join, anti_join
              Binds (no by=): bind_rows, bind_cols
              Sets (no by=): intersect, union, setdiff
        add: Columns to select from y (optional, for joins). If None, uses all columns.
        suffix: Suffix for overlapping column names in y. Default: '_right'

    Returns:
        pl.DataFrame: Combined dataset

    Examples:
        >>> rsm.eda.combine(orders, customers, by='customer_id')  # Inner join
        >>> rsm.eda.combine(orders, customers, by='customer_id', type='left_join')
        >>> rsm.eda.combine(df1, df2, type='bind_rows')  # Stack vertically
        >>> rsm.eda.combine(df1, df2, type='intersect')  # Common rows
    """
    # Validate type
    if type not in ALL_TYPES:
        raise ValueError(
            f"Unknown combine type: {type}\n"
            f"Supported: {', '.join(sorted(ALL_TYPES))}"
        )

    is_join = type in JOIN_TYPES

    # Validate by= for joins
    if is_join and not by:
        raise ValueError(
            f"Join type '{type}' requires by= parameter.\n"
            "Example: rsm.eda.combine(x, y, by='customer_id', type='inner_join')"
        )

    # Convert to LazyFrame for consistency
    x_lf = x.lazy() if isinstance(x, pl.DataFrame) else x
    y_lf = y.lazy() if isinstance(y, pl.DataFrame) else y

    # Optionally limit columns from y
    if add and is_join:
        left_on, right_on = _parse_by(by)
        # Include join keys + add columns
        keep_cols = list(set(right_on + add))
        y_lf = y_lf.select(keep_cols)

    # Execute based on type
    if is_join:
        left_on, right_on = _parse_by(by)
        how = POLARS_JOIN_MAP[type]

        if left_on == right_on:
            result = x_lf.join(y_lf, on=left_on, how=how, suffix=suffix)
        else:
            result = x_lf.join(y_lf, left_on=left_on, right_on=right_on,
                               how=how, suffix=suffix)

    elif type == "bind_rows":
        result = pl.concat([x_lf, y_lf], how="diagonal")

    elif type == "bind_cols":
        # Horizontal concat - collect for frame alignment
        x_df = x_lf.collect()
        y_df = y_lf.collect()
        result = pl.concat([x_df, y_df], how="horizontal").lazy()

    elif type == "intersect":
        # Rows present in both datasets
        x_df = x_lf.collect()
        y_df = y_lf.collect()
        result = x_df.join(y_df, on=x_df.columns, how="semi").lazy()

    elif type == "union":
        # All rows from both, deduplicated
        result = pl.concat([x_lf, y_lf]).unique()

    elif type == "setdiff":
        # Rows in x but not in y
        x_df = x_lf.collect()
        y_df = y_lf.collect()
        result = x_df.join(y_df, on=x_df.columns, how="anti").lazy()

    return result.collect()
