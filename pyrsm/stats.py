import pandas as pd
import polars as pl
from math import sqrt
from pyrsm.utils import ifelse, setdiff, intersect


def varprop(x, na: bool = True) -> float:
    """
    Calculate the variance for a proportion

    Parameters
    ----------
    x : list, pl.Series, pd.Series, or array-like
        Numeric variable with only values 0 and 1
    na : bool
        Drop missing values before calculating (True or False)

    Returns
    -------
    float
        Calculated variance for a proportion based on a vector of 0 and 1 values

    Examples
    --------
    varprop([0, 1, 1, 1, 0, 0, 0])
    """
    if isinstance(x, pd.Series):
        x = pl.from_pandas(x)
    elif not isinstance(x, pl.Series):
        x = pl.Series(x)

    if na:
        p = x.drop_nulls().mean()
    else:
        p = x.mean()

    return p * (1 - p)


def sdprop(x, na: bool = True) -> float:
    """
    Calculate the standard deviation for a proportion

    Parameters
    ----------
    x : list, pl.Series, pd.Series, or array-like
        Numeric variable with only values 0 and 1
    na : bool
        Drop missing values before calculating (True or False)

    Returns
    -------
    float
        Calculated standard deviation for a proportion based on a vector of 0 and 1 values

    Examples
    --------
    sdprop([0, 1, 1, 1, 0, 0, 0])
    """
    return sqrt(varprop(x, na=na))


def seprop(x, na: bool = True) -> float:
    """
    Calculate the standard error for a proportion

    Parameters
    ----------
    x : list, pl.Series, pd.Series, or array-like
        Numeric variable with only values 0 and 1
    na : bool
        Drop missing values before calculating (True or False)

    Returns
    -------
    float
        Calculated standard error for a proportion based on a vector of 0 and 1 values

    Examples
    --------
    seprop([0, 1, 1, 1, 0, 0, 0])
    """
    if isinstance(x, pd.Series):
        x = pl.from_pandas(x)
    elif not isinstance(x, pl.Series):
        x = pl.Series(x)

    if na:
        x = x.drop_nulls()

    n = x.len()
    return sqrt(varprop(x, na=False) / n)


def _convert_to_polars(df, wt):
    """Helper to convert inputs to polars DataFrame and Series."""
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if isinstance(wt, (pd.Series, list)):
        wt = pl.Series("__wt__", list(wt))
    elif not isinstance(wt, pl.Series):
        wt = pl.Series("__wt__", list(wt))

    return df, wt


def weighted_mean(df, wt) -> pl.Series:
    """
    Calculate the weighted mean for a DataFrame

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        All columns in the dataframe are expected to be numeric
    wt : list, pl.Series, pd.Series, or array-like
        Weights to use during calculation. Length must match number of rows in df

    Returns
    -------
    pl.Series
        Series of weighted means for each column in df

    Examples
    --------
    df = pl.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0]})
    wt = [1, 10, 1, 10, 1, 10, 1]
    weighted_mean(df, wt)
    """
    df, wt = _convert_to_polars(df, wt)
    total_weight = wt.sum()
    result = df.select([
        (pl.col(col) * wt).sum() / total_weight for col in df.columns
    ])
    return pl.Series(result.row(0))


def weighted_sd(df, wt, ddof: int = 0) -> pl.Series:
    """
    Calculate the weighted standard deviation for a DataFrame

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        All columns in the dataframe are expected to be numeric
    wt : list, pl.Series, pd.Series, or array-like
        Weights to use during calculation. Length must match number of rows in df
    ddof : int, default 0
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. Default 0 matches numpy/sklearn.

    Returns
    -------
    pl.Series
        Series of weighted standard deviations for each column in df

    Examples
    --------
    df = pl.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0]})
    wt = [1, 10, 1, 10, 1, 10, 1]
    weighted_sd(df, wt)
    """
    df, wt = _convert_to_polars(df, wt)
    total_weight = wt.sum()
    wts = wt / (total_weight - ddof)  # weights for variance
    means = weighted_mean(df, wt)

    # Calculate weighted std: sqrt(sum(w * (x - mean)^2))
    result = df.select([
        ((wts * (pl.col(col) - means[i]) ** 2).sum().sqrt()).alias(col)
        for i, col in enumerate(df.columns)
    ])
    return pl.Series(result.row(0))


def scale_df(
    df,
    wt=None,
    sf: int = 1,
    excl: list = None,
    train=None,
    ddof: int = 0,
    stats: bool = False,
    means: dict = None,
    stds: dict = None,
):
    """
    Scale the numeric variables in a DataFrame

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        DataFrame with numeric variables
    wt : pl.Series, pd.Series, list, or None
        Weights to use during scaling. Length must match number of rows in df
    sf : int
        Scale factor to use. Default is 1 standard deviation but 2 standard deviation is
        also a good option to enhance comparability with categorical variables
    excl : None or list
        Provide list of column names to exclude when applying standardization
    train : pl.Series, pd.Series, list, or None
        A series of True and False values. Values that are True are
        used to calculate the mean and standard deviation
    stats : bool
        Return the mean and standard deviation for each column
    ddof : int, default 0
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. Default 0 matches numpy/sklearn.
    means : None or dict
        Means to apply for (re)scaling variables
    stds : None or dict
        Standard deviations to apply for (re)scaling variables

    Returns
    -------
    DataFrame with all numeric variables standardized. If stats is True,
    a tuple of (DataFrame, means dict, stds dict) is returned.
    Returns same type (pandas or polars) as input.

    Examples
    --------
    df = pl.DataFrame({"x": [0, 1, 1, 1, 0, 3, 0]})
    df = scale_df(df)
    """
    # Track input type to return same type
    input_is_pandas = isinstance(df, pd.DataFrame)

    # Convert to polars for processing
    if input_is_pandas:
        df = pl.from_pandas(df)

    if means is not None and stds is not None:
        is_num = list(means.keys())
    else:
        # Find numeric columns that aren't binary (0/1)
        is_num = []
        for col in df.columns:
            dtype = df[col].dtype
            if dtype.is_numeric():
                col_data = df[col]
                is_binary = (
                    col_data.n_unique() == 2
                    and col_data.min() == 0
                    and col_data.max() == 1
                )
                if not is_binary:
                    is_num.append(col)

    is_num = intersect(is_num, df.columns)
    if excl is not None:
        is_num = setdiff(is_num, excl)

    if len(is_num) == 0:
        result = df.to_pandas() if input_is_pandas else df
        if stats:
            return result, {}, {}
        return result

    # Cast numeric columns to Float64
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in is_num])

    # Select only numeric columns for calculations
    dfs = df.select(is_num)

    # Create training mask
    if train is None:
        train_mask = pl.Series([True] * df.height)
    elif isinstance(train, (pd.Series, list)):
        train_mask = pl.Series(list(train))
    else:
        train_mask = train

    # Filter to training data
    dfs_train = dfs.filter(train_mask)

    if wt is None:
        # Unweighted mean and std
        if means is None:
            means_list = [dfs_train[c].mean() for c in is_num]
        else:
            means_list = [means[c] for c in is_num]

        if stds is None:
            stds_list = [sf * dfs_train[c].std(ddof=ddof) for c in is_num]
        else:
            stds_list = [stds[c] for c in is_num]
    else:
        # Weighted mean and std - convert wt to polars Series
        if isinstance(wt, (pd.Series, list)):
            wt = pl.Series("__wt__", list(wt))
        elif not isinstance(wt, pl.Series):
            wt = pl.Series("__wt__", list(wt))

        wt_train = wt.filter(train_mask)

        if means is None:
            means_list = weighted_mean(dfs_train, wt_train)
        else:
            means_list = [means[c] for c in is_num]

        if stds is None:
            stds_list = [sf * s for s in weighted_sd(dfs_train, wt_train, ddof=ddof)]
        else:
            stds_list = [stds[c] for c in is_num]

    # Apply standardization: (x - mean) / std
    scaled_exprs = [
        ((pl.col(c) - means_list[i]) / stds_list[i]).alias(c) for i, c in enumerate(is_num)
    ]

    # Get non-numeric columns to preserve
    other_cols = [c for c in df.columns if c not in is_num]

    # Combine: keep other columns, replace numeric with scaled
    if other_cols:
        df = df.select(other_cols + scaled_exprs)
    else:
        df = df.select(scaled_exprs)

    # Convert back to pandas if input was pandas
    if input_is_pandas:
        df = df.to_pandas()

    if stats:
        means_dict = {c: means_list[i] for i, c in enumerate(is_num)}
        stds_dict = {c: stds_list[i] for i, c in enumerate(is_num)}
        return df, means_dict, stds_dict
    else:
        return df
