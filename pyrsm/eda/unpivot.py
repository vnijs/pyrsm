"""
unpivot() - Convert wide format to long format (reverse of pivot).

Examples:
    import pyrsm as rsm

    # Basic unpivot - specify columns to unpivot
    rsm.eda.unpivot(df, on=['Q1', 'Q2', 'Q3'])

    # With ID columns to keep
    rsm.eda.unpivot(df, on=['Q1', 'Q2', 'Q3'], id_vars='name')

    # Custom column names
    rsm.eda.unpivot(df, on=['Q1', 'Q2'], id_vars='name', variable_name='quarter', value_name='sales')
"""

from typing import List, Optional, Union
import polars as pl


def unpivot(
    df: Union[pl.DataFrame, pl.LazyFrame],
    on: Optional[Union[str, List[str]]] = None,
    id_vars: Optional[Union[str, List[str]]] = None,
    variable_name: str = "variable",
    value_name: str = "value",
) -> pl.DataFrame:
    """
    Convert wide format data to long format (reverse of pivot).

    Args:
        df: Polars DataFrame or LazyFrame
        on: Column(s) to unpivot. If None, uses all columns not in id_vars
        id_vars: Column(s) to keep as identifier variables
        variable_name: Name for the new variable column. Default: 'variable'
        value_name: Name for the new value column. Default: 'value'

    Returns:
        DataFrame in long format

    Examples:
        >>> # Wide format: name, Q1, Q2, Q3, Q4
        >>> rsm.eda.unpivot(df, on=['Q1', 'Q2', 'Q3', 'Q4'], id_vars='name')
        >>> # Long format: name, variable, value

        >>> rsm.eda.unpivot(df, on=['Q1', 'Q2'], id_vars='name')
        >>> rsm.eda.unpivot(df, on=['Q1', 'Q2'], id_vars='name', variable_name='quarter', value_name='sales')
    """
    # Convert to DataFrame if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Normalize id_vars to list
    if id_vars is None:
        id_vars_list = []
    elif isinstance(id_vars, str):
        id_vars_list = [id_vars]
    else:
        id_vars_list = list(id_vars)

    # Normalize on to list (if specified)
    if on is None:
        value_vars_list = None
    elif isinstance(on, str):
        value_vars_list = [on]
    else:
        value_vars_list = list(on)

    # Call Polars unpivot
    result = df.unpivot(
        on=value_vars_list,
        index=id_vars_list,
        variable_name=variable_name,
        value_name=value_name,
    )

    return result
