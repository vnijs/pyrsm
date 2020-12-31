import numpy as np
import pandas as pd
from itertools import product
from datetime import date, datetime
from math import ceil


def add_description(df, md="", path=""):
    """
    Add a description to a Pandas dataframe in markdown format

    Parameters
    ----------
    df : Pandas dataframe
    md : str
        Data description in markdown format
    path : str
        Path to a text file with the data description in markdown format

    Returns
    -------
    Pandas dataframe with added description
    """

    if path != "":
        f = open(path, "r")
        md = f.read()
        f.close()
    elif md == "":
        print("Provide either text (markdown) or the path to a file")
        print("with the data description")

    df._metadata.append("description")
    df.description = md
    return df


def describe(df):
    """Print out Pandas dataframe description attribute if available. Else use Pandas' description method to provide summary statistics"""
    if hasattr(df, "description"):
        print(df.description)
    else:
        print("No description attribute available")
        return df.describe()


def ifelse(cond, if_true, if_false):
    """
    Oneline if-else function like R

    Parameters
    ----------
    cond : List, pandas series, or numpy array of boolean values
    if_true : float, int, str, or list, pandas series, or numpy array
        Value to use if the condition is True
    if_false : float, int, str, or list, pandas series, or numpy array
        Value to use if the condition is False

    Returns
    -------
    Numpy array if the length of cond > 1. Else the same object type as either if_true or if_false

    Examples
    --------
    ifelse(2 > 3, "greater", "smaller")
    ifelse(np.array([2, 3, 4]) > 2, 1, 0)
    ifelse(np.array([2, 3, 4]) > 2, np.array([-1, -2, -3]), np.array([1, 2, 3]))
    """
    try:
        len(cond) > 1  # catching "TypeError: object of type 'bool' has no len()"
        return np.where(cond, if_true, if_false)
    except TypeError:
        if cond:
            return if_true
        else:
            return if_false


def format_nr(x, sym="", dec=2, perc=False):
    """
    Format a number or numeric vector with a specified number of decimal places,
    thousand sep, and a symbol

    Parameters
    ----------
    x : numeric (vector)
        Number or vector to format
    sym : str
        Symbol to use
    dec : int
        Number of decimals to show
    perc : boolean
        Display numbers as a percentage

    Returns
    -------
    str
        Number(s) in the desired format
    """
    try:
        len(x) > 1  # catching "TypeError: object of type 'bool' has no len()"
        return [format_nr(i, sym, dec, perc) for i in x]
    except TypeError:
        if dec == 0:
            x = int(x)

        if perc:
            return sym + "{:,}".format(round((100.0 * x), dec)) + "%"
        else:
            return sym + "{:,}".format(round((x), dec))


def levels_list(df):
    """
    Provide a pandas dataframe and get a dictionary back with the unique values for
    each column

    Parameters
    ----------
    df: Pandas dataframe

    Returns
    -------
    dict
        Dictionary with unique values (levels) for each column

    Examples
    --------
    df = pd.DataFrame({
        "var1": ["a", "b", "a"],
        "var2": [1, 2, 1]
    })
    levels_list(df)
    """
    return {str(col): list(df[col].unique()) for col in df.columns}


def expand_grid(dct):
    """
    Provide a dictionary and get a pandas dataframe back with all possible
    value combinations

    Parameters
    ----------
    dct
        Dictionary with value combinations to expand

    Returns
    -------
    Pandas dataframe with all possible value combination

    Example
    -------
    expand_grid({"var1": ["a", "b"], "var2": [1, 2]})
    """
    return pd.DataFrame([val for val in product(*dct.values())], columns=dct.keys())


def table2data(df, freq):
    """
    Provide a pandas dataframe and get dataframe back with all
    the total number of rows equal to the sum of the frequency
    variable

    Parameters
    ----------
    df: Pandas dataframe
    freq: str
        String with the variable name of the frequency column in df

    Returns
    -------
    Pandas dataframe expanded in size based on the frequencies in selected column

    Examples
    --------
    df = pd.DataFrame({"var1": ["a", "b", "a"], "freq": [5, 2, 3]})
    table2data(df, "freq")
    """

    return df.loc[df.index.repeat(df[freq])]


def setdiff(x, y):
    """
    Returns a numpy array of unique elements in x that are not in y

    Parameters
    ----------

    x : List type or that can be converted to a list
    y : List type or that can be converted to a list

    Returns
    -------
    list
        Elements in x that are not in y

    Examples
    --------
    setdiff(["a", "b", "c"], ["b", "x"])
    """

    x = np.unique(np.array(x))
    y = np.unique(np.array(y))
    return list(np.setdiff1d(x, y, assume_unique=True))


def union(x, y):
    """
    Return the unique, sorted array of values that are in either of the two input arrays

    Parameters
    ----------

    x : List type or that can be converted to a list
    y : List type or that can be converted to a list

    Returns
    -------
    list
        Unique, sorted array of values that are in either of the two input arrays

    Examples
    --------
    union(["a", "b", "c"], ["b", "x"])
    """

    x = np.array(x)
    y = np.array(y)
    return list(np.union1d(x, y))


def intersect(x, y):
    """
    Return the unique, sorted array of values that are in both input arrays

    Parameters
    ----------

    x : List type or that can be converted to a list
    y : List type or that can be converted to a list

    Returns
    -------
    list
        Unique, sorted array of values that are in both input arrays

    Examples
    --------
    intersect(["a", "b", "c"], ["b", "x"])
    """
    x = np.array(x)
    y = np.array(y)
    return list(np.intersect1d(x, y))


def lag(arr, num=1, fill=np.nan):
    """
    Create a numpy array of time series data, shifted ahead or back a number of periods

    Parameters
    ----------
    arr : Numpy array
        A numpy array shift ahead or back
    num : int
        Number of periods to shift values in arr. For example, num=1 would
        lag the data by one period, while -1 would create a one period lead
    fill : Optional
        Input to use to replace missing values created by shifting
        the values in arr forwards of backwards

    Returns
    -------
    Numpy array containing a shifted copy of arr

    Adapted from https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill
        result[:num] = arr[-num:]
    else:
        result = arr.copy()
    return result


def lead(arr, num=1, fill=np.nan):
    """Convenience function for leading periods. Uses the lag function in pyrsm"""
    return lag(arr, num=-num, fill=fill)


def months_abb(start=1, nr=12, year=datetime.today().year):
    """
    Create a list of abbreviated month labels

    Parameters
    ----------
    start : int
        Numeric value of the first month in the list (e.g., January is 1)
    nr : int
        Number of months to include in the list
    year : int
        Input to use to replace missing values created by shifting
        the values in arr forwards of backwards

    Returns
    -------
    list
        List of abbreviated month labels
    """

    rng = ceil((nr + (start - 1)) / 12)
    mnths = [
        date(year, m, 1).strftime("%B")[0:3]
        for i in range(1, rng + 1)
        for m in range(1, 13)
    ]
    start -= 1
    return mnths[start : (nr + start)]
