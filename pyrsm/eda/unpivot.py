"""
unpivot() - Convert wide format to long format (reverse of pivot).

Examples:
    import pyrsm as rsm

    # Basic unpivot - all columns except ID become values
    rsm.eda.unpivot(df, id_vars='name')

    # Specify which columns to unpivot
    rsm.eda.unpivot(df, id_vars='name', value_vars=['Q1', 'Q2', 'Q3'])

    # Custom column names
    rsm.eda.unpivot(df, id_vars='name', variable_name='quarter', value_name='sales')
"""

from typing import List, Optional, Union
import polars as pl


def unpivot(
    df: Union[pl.DataFrame, pl.LazyFrame],
    id_vars: Optional[Union[str, List[str]]] = None,
    value_vars: Optional[Union[str, List[str]]] = None,
    variable_name: str = "variable",
    value_name: str = "value",
) -> pl.DataFrame:
    """
    Convert wide format data to long format (reverse of pivot).

    Args:
        df: Polars DataFrame or LazyFrame
        id_vars: Column(s) to keep as identifier variables
        value_vars: Column(s) to unpivot. If None, uses all columns not in id_vars
        variable_name: Name for the new variable column. Default: 'variable'
        value_name: Name for the new value column. Default: 'value'

    Returns:
        DataFrame in long format

    Examples:
        >>> # Wide format: name, Q1, Q2, Q3, Q4
        >>> rsm.eda.unpivot(df, id_vars='name')
        >>> # Long format: name, variable, value

        >>> rsm.eda.unpivot(df, id_vars='name', value_vars=['Q1', 'Q2'])
        >>> rsm.eda.unpivot(df, id_vars='name', variable_name='quarter', value_name='sales')
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

    # Normalize value_vars to list (if specified)
    if value_vars is None:
        value_vars_list = None
    elif isinstance(value_vars, str):
        value_vars_list = [value_vars]
    else:
        value_vars_list = list(value_vars)

    # Call Polars unpivot
    result = df.unpivot(
        on=value_vars_list,
        index=id_vars_list,
        variable_name=variable_name,
        value_name=value_name,
    )

    return result
