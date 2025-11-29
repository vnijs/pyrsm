"""
rsm.eda - Exploratory Data Analysis functions for Polars DataFrames.

Provides clean Python API for common EDA operations:
- explore(): Summary statistics for numeric columns
- pivot(): Pivot tables and crosstabs
- combine(): Combine datasets using joins, binds, or set operations
"""

from .explore import explore
from .pivot import pivot
from .combine import combine

__all__ = ["explore", "pivot", "combine"]
