import numpy as np
import pandas as pd
from itertools import product


def add_description(df, md="", path=""):
    """
    Add a description to a pandas data frame

    Arguments:
    df      A pandas data frame
    md      Data description in markdown format
    path    Path to a text file with the data description in markdown format

    Return:
    Pandas data frame with added description
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
    """Print out data frame description attribute if available"""
    if hasattr(df, "description"):
        print(df.description)
    else:
        print("No description attribute available")
        return df.describe()


def ifelse(cond, if_true, if_false):
    """Oneline if-else like R"""
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

    Return
    ------
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

    Return
    ------
    dict
        Dictionary with unique values (levels) for each column
    
    Example
    -------
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
    dict
        Dictionary with value combinations to expand

    Return
    ------
    DataFrame
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
    freq: String with the variable name of the frequency column
        in df

    Return
    ------
    DataFrame
        Pandas dataframe expanded in size based on the frequencies
        in selected column

    Example
    -------
    df = pd.DataFrame({"var1": ["a", "b", "a"], "freq": [5, 2, 3]})
    table2data(df, "freq")
    """
    return df.loc[df.index.repeat(df[freq])]


def setdiff(x, y):
    """
    Returns a numpy array elements in x that are not in y

    x and y: Inputs of a list type or that can be converted to a list

    Example
    -------
    setdiff(["a", "b", "c"], ["b", "x"])
    """
    x = np.unique(np.array(x))
    y = np.unique(np.array(y))
    return list(np.setdiff1d(x, y, assume_unique=True))


def union(x, y):
    """
    Return the unique, sorted array of values that are in either of the two input
        arrays

    x and y: Inputs of a list type or that can be converted to a list

    Example
    -------
    union(["a", "b", "c"], ["b", "x"])
    """
    x = np.array(x)
    y = np.array(y)
    return list(np.union1d(x, y))


def intersect(x, y):
    """
    Return the unique, sorted array of values that are in both input arrays

    x and y: Inputs of a list type or that can be converted to a list

    Example
    -------
    intersect(["a", "b", "c"], ["b", "x"])
    """
    x = np.array(x)
    y = np.array(y)
    return list(np.intersect1d(x, y))

