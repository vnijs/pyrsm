"""
combine() - Combine datasets using joins, binds, or set operations.

Uses Polars-native parameter names (on, how) for familiarity.

Examples:
    import pyrsm as rsm

    # Join datasets (Polars-style)
    rsm.eda.combine(orders, customers, on='customer_id')  # inner join (default)
    rsm.eda.combine(orders, customers, on='customer_id', how='left')

    # Different key names
    rsm.eda.combine(orders, customers, left_on='cust_id', right_on='customer_id')

    # Bind operations
    rsm.eda.combine(df1, df2, how='bind_rows')
    rsm.eda.combine(df1, df2, how='bind_cols')

    # Set operations
    rsm.eda.combine(df1, df2, how='intersect')
"""

from typing import List, Optional, Union
import polars as pl

# All supported combine types
JOIN_TYPES = {"inner", "left", "right", "full", "semi", "anti"}
BIND_TYPES = {"bind_rows", "bind_cols"}
SET_TYPES = {"intersect", "union", "setdiff"}
ALL_TYPES = JOIN_TYPES | BIND_TYPES | SET_TYPES


def _align_join_key_dtypes(
    x_lf: pl.LazyFrame,
    y_lf: pl.LazyFrame,
    left_on: List[str],
    right_on: List[str],
) -> tuple:
    """
    Align join key dtypes between two LazyFrames.

    Smart dtype alignment preserving the "better" type:
    - Cat vs Str: Cast Str → Cat (preserves categorical benefits)
    - Int vs Float: Cast Int → Float (lossless upcast)
    - Other mismatches: Cast to String as fallback

    Returns:
        Tuple of (x_lf, y_lf) with aligned key dtypes
    """
    x_schema = x_lf.collect_schema()
    y_schema = y_lf.collect_schema()

    x_casts = []
    y_casts = []

    # Numeric types for Int vs Float detection
    int_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    float_types = {pl.Float32, pl.Float64}

    for left_col, right_col in zip(left_on, right_on):
        left_dtype = x_schema.get(left_col)
        right_dtype = y_schema.get(right_col)

        if left_dtype is None or right_dtype is None:
            continue  # Column doesn't exist, let Polars raise the error

        if left_dtype == right_dtype:
            continue  # Already aligned

        # Case 1: Cat vs Str - cast Str to Cat (preserves categorical benefits)
        if left_dtype == pl.Categorical and right_dtype in (pl.String, pl.Utf8):
            y_casts.append(pl.col(right_col).cast(pl.Categorical))
        elif right_dtype == pl.Categorical and left_dtype in (pl.String, pl.Utf8):
            x_casts.append(pl.col(left_col).cast(pl.Categorical))

        # Case 2: Int vs Float - cast Int to Float (lossless upcast)
        elif left_dtype in int_types and right_dtype in float_types:
            x_casts.append(pl.col(left_col).cast(right_dtype))
        elif right_dtype in int_types and left_dtype in float_types:
            y_casts.append(pl.col(right_col).cast(left_dtype))

        # Case 3: Fallback - cast both to String
        else:
            if left_dtype != pl.String:
                x_casts.append(pl.col(left_col).cast(pl.String))
            if right_dtype != pl.String:
                y_casts.append(pl.col(right_col).cast(pl.String))

    # Apply casts
    if x_casts:
        x_lf = x_lf.with_columns(x_casts)
    if y_casts:
        y_lf = y_lf.with_columns(y_casts)

    return x_lf, y_lf


def combine(
    x: Union[pl.DataFrame, pl.LazyFrame],
    y: Union[pl.DataFrame, pl.LazyFrame],
    on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    add: Optional[List[str]] = None,
    suffix: str = "_right",
) -> pl.DataFrame:
    """
    Combine two datasets using joins, binds, or set operations.

    Uses Polars-native parameter names for familiarity.

    Args:
        x: Left/first dataset (Polars DataFrame or LazyFrame)
        y: Right/second dataset to combine with x
        on: Join key(s) when same column name on both sides.
            Can be string ("id") or list (["id", "date"]).
        how: Combine type. Default: 'inner'
             Joins: inner, left, right, full, semi, anti
             Binds: bind_rows, bind_cols
             Sets: intersect, union, setdiff
        left_on: Join key(s) from x when names differ from y.
        right_on: Join key(s) from y when names differ from x.
        add: Columns to select from y (optional, for joins). If None, uses all.
        suffix: Suffix for overlapping column names. Default: '_right'

    Returns:
        pl.DataFrame: Combined dataset

    Notes:
        - Join key dtypes are automatically aligned (e.g., Cat cast to Str)

    Examples:
        >>> rsm.eda.combine(orders, customers, on='customer_id')  # Inner join
        >>> rsm.eda.combine(orders, customers, on='customer_id', how='left')
        >>> rsm.eda.combine(orders, customers, left_on='cust_id', right_on='customer_id')
        >>> rsm.eda.combine(df1, df2, how='bind_rows')  # Stack vertically
        >>> rsm.eda.combine(df1, df2, how='intersect')  # Common rows
    """
    # Validate how
    if how not in ALL_TYPES:
        raise ValueError(
            f"Unknown combine type: {how}\n"
            f"Supported: {', '.join(sorted(ALL_TYPES))}"
        )

    is_join = how in JOIN_TYPES

    # Validate join keys
    if is_join:
        if on is None and (left_on is None or right_on is None):
            raise ValueError(
                f"Join type '{how}' requires on= or (left_on= and right_on=).\n"
                "Example: rsm.eda.combine(x, y, on='customer_id', how='left')"
            )

    # Convert to LazyFrame for consistency
    x_lf = x.lazy() if isinstance(x, pl.DataFrame) else x
    y_lf = y.lazy() if isinstance(y, pl.DataFrame) else y

    # Normalize join keys to lists
    if on is not None:
        left_keys = [on] if isinstance(on, str) else list(on)
        right_keys = left_keys
    elif left_on is not None and right_on is not None:
        left_keys = [left_on] if isinstance(left_on, str) else list(left_on)
        right_keys = [right_on] if isinstance(right_on, str) else list(right_on)
    else:
        left_keys = []
        right_keys = []

    # Optionally limit columns from y
    if add and is_join:
        keep_cols = list(set(right_keys + add))
        y_lf = y_lf.select(keep_cols)

    # Execute based on type
    if is_join:
        # Auto-align join key dtypes
        x_lf, y_lf = _align_join_key_dtypes(x_lf, y_lf, left_keys, right_keys)

        if left_keys == right_keys:
            result = x_lf.join(y_lf, on=left_keys, how=how, suffix=suffix)
        else:
            result = x_lf.join(y_lf, left_on=left_keys, right_on=right_keys,
                               how=how, suffix=suffix)

    elif how == "bind_rows":
        result = pl.concat([x_lf, y_lf], how="diagonal")

    elif how == "bind_cols":
        # Horizontal concat - collect for frame alignment
        x_df = x_lf.collect()
        y_df = y_lf.collect()
        result = pl.concat([x_df, y_df], how="horizontal").lazy()

    elif how == "intersect":
        # Rows present in both datasets
        x_df = x_lf.collect()
        y_df = y_lf.collect()
        result = x_df.join(y_df, on=x_df.columns, how="semi").lazy()

    elif how == "union":
        # All rows from both, deduplicated
        result = pl.concat([x_lf, y_lf]).unique()

    elif how == "setdiff":
        # Rows in x but not in y
        x_df = x_lf.collect()
        y_df = y_lf.collect()
        result = x_df.join(y_df, on=x_df.columns, how="anti").lazy()

    return result.collect()
