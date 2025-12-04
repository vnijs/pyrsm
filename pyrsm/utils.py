import inspect
import json
from datetime import date, datetime
from itertools import product
from math import ceil
from sys import modules

import numpy as np
import pandas as pd
import polars as pl
from IPython.display import Markdown, display


def add_description(df, md="", path=""):
    """
    Add a description to a Pandas DataFrame in markdown format

    Parameters
    ----------
    df : Pandas DataFrame
    md : str
        Data description in markdown format
    path : str
        Path to a text file with the data description in markdown format

    Returns
    -------
    Pandas DataFrame with added description
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


def describe(df, prn=True):
    """
    Print out Pandas DataFrame description attribute if available. Else use Pandas'
    description method to provide summary statistics
    """
    if hasattr(df, "description"):
        if "ipykernel" in modules and prn:
            display(Markdown(df.description))
        elif prn:
            print(df.description)
        else:
            return df.description
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
        Number of decimal places to use in rounding
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
    Provide a DataFrame and get a dictionary back with the unique values for
    each column

    Parameters
    ----------
    df: Pandas or Polars DataFrame

    Returns
    -------
    dict
        Dictionary with unique values (levels) for each column

    Examples
    --------
    df = pl.DataFrame({
        "var1": ["a", "b", "a"],
        "var2": [1, 2, 1]
    })
    levels_list(df)
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    return {str(col): df[col].unique().to_list() for col in df.columns}


def expand_grid(dct, schema=None):
    """
    Provide a dictionary and get a polars DataFrame back with all possible
    value combinations.

    Parameters
    ----------
    dct : dict
        Dictionary with value combinations to expand
    schema : dict of polars dtypes, optional
        Column types to cast the result to (must be polars dtypes)

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with all possible value combinations

    Example
    -------
    expand_grid({"var1": ["a", "b"], "var2": [1, 2]})
    """
    rows = list(product(*dct.values()))
    df = pl.DataFrame(rows, schema=list(dct.keys()), orient="row")

    if schema is not None:
        # Cast columns to match original schema if provided
        cast_exprs = []
        for col in df.columns:
            if col in schema:
                target_dtype = schema[col]
                current_dtype = df[col].dtype
                # Only cast if types differ and target is a valid polars dtype
                if current_dtype != target_dtype and isinstance(target_dtype, pl.DataType):
                    cast_exprs.append(pl.col(col).cast(target_dtype))
                else:
                    cast_exprs.append(pl.col(col))
            else:
                cast_exprs.append(pl.col(col))
        df = df.select(cast_exprs)

    return df


def table2data(df, freq):
    """
    Provide a DataFrame and get a DataFrame back with the total number of rows
    equal to the sum of the frequency variable

    Parameters
    ----------
    df: Pandas or Polars DataFrame
    freq: str
        String with the variable name of the frequency column in df

    Returns
    -------
    pl.DataFrame
        Polars DataFrame expanded in size based on the frequencies in selected column

    Examples
    --------
    df = pl.DataFrame({"var1": ["a", "b", "a"], "freq": [5, 2, 3]})
    table2data(df, "freq")
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # Get columns excluding the frequency column
    other_cols = [c for c in df.columns if c != freq]

    # Repeat each row by its frequency value
    return df.select(other_cols).select(
        pl.all().repeat_by(df[freq])
    ).explode(pl.all())


def setdiff(x, y, sort=False):
    """
    Returns unique elements in x that are not in y

    Parameters
    ----------
    x : List or iterable
    y : List or iterable
    sort : boolean
        Sort the output

    Returns
    -------
    list
        Elements in x that are not in y

    Examples
    --------
    setdiff(["a", "b", "c"], ["b", "x"])
    """
    result = list(dict.fromkeys(item for item in x if item not in set(y)))
    return sorted(result) if sort else result


def union(x, y):
    """
    Return the unique, sorted list of values that are in either of the two input lists

    Parameters
    ----------
    x : List or iterable
    y : List or iterable

    Returns
    -------
    list
        Unique, sorted list of values that are in either input

    Examples
    --------
    union(["a", "b", "c"], ["b", "x"])
    """
    return sorted(set(x) | set(y))


def intersect(x, y):
    """
    Return the unique, sorted list of values that are in both input lists

    Parameters
    ----------
    x : List or iterable
    y : List or iterable

    Returns
    -------
    list
        Unique, sorted list of values that are in both inputs

    Examples
    --------
    intersect(["a", "b", "c"], ["b", "x"])
    """
    return sorted(set(x) & set(y))


def lag(arr, num=1, fill=None):
    """
    Shift data by a number of periods (lag)

    Parameters
    ----------
    arr : list, pl.Series, or array-like
        Data to shift
    num : int
        Number of periods to shift. Positive values lag (shift forward),
        negative values lead (shift backward)
    fill : Optional
        Value to use for missing values created by shifting. Default is None (null)

    Returns
    -------
    pl.Series
        Shifted data as a polars Series

    Examples
    --------
    lag([1, 2, 3, 4], num=1)  # [null, 1, 2, 3]
    lag([1, 2, 3, 4], num=-1)  # [2, 3, 4, null]
    """
    if isinstance(arr, pl.Series):
        s = arr
    else:
        s = pl.Series(arr)

    result = s.shift(num)
    if fill is not None:
        result = result.fill_null(fill)
    return result


def lead(arr, num=1, fill=None):
    """
    Shift data by a number of periods (lead)

    Convenience function for leading periods. Uses the lag function with negative num.

    Parameters
    ----------
    arr : list, pl.Series, or array-like
        Data to shift
    num : int
        Number of periods to lead (shift backward)
    fill : Optional
        Value to use for missing values. Default is None (null)

    Returns
    -------
    pl.Series
        Shifted data as a polars Series
    """
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
    mnths = [date(year, m, 1).strftime("%B")[0:3] for i in range(1, rng + 1) for m in range(1, 13)]
    start -= 1
    return mnths[start : (nr + start)]


def md(x: str) -> None:
    """
    Use in-line python code to generate markdown output

    Parameters
    ----------
    x : A python f-string or the path to a markdown file

    Returns
    -------
    None - Markdown output is printed

    Examples
    --------
    md(f"### In-line code to markdown results")
    radius = 10
    md(f"The radius of the circle is {radius}.")
    md("./path-to-markdown-file.md")
    """
    display(Markdown(x))


def md_notebook(nb: str, type="python") -> None:
    """
    Print code from another notebook in markdown format
    This can be useful when you are using the %run magick
    in a notebook to source other notebooks. This way you
    both the notebook output and the code

    Parameters
    ----------
    nb : Path to a Jupyter Notebook file

    Returns
    -------
    None - Markdown output is printed

    Examples
    --------
    md("./path-to-notebook-file.ipynb")
    """
    with open(nb) as f:
        data = json.load(f)

    md_return = "\n"

    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            md_return += f"```{type}\n" + "".join(cell["source"]) + "\n```\n"
        elif cell["cell_type"] == "markdown":
            md_return += "\n".join(cell["source"]) + "\n"
        else:
            md_return += "\n"

    md(md_return)


def odir(obj, private: bool = False) -> dict:
    """
    List an objects attributes and 'public' methods

    Parameters
    ----------
    obj : Any python object
    private : Boolean, default is false to exclude 'private' methods and attributes

    Returns
    -------
    Dictionary with names of attributes and methods

    Examples
    --------
    odir(["a"])
    """
    mth = []
    attr = []
    for i in inspect.getmembers(obj):
        if private or not i[0].startswith("_"):
            if inspect.ismethod(i[1]) or inspect.isbuiltin(i[1]):
                mth.append(i[0])
            else:
                attr.append(i[0])

    return {"methods": mth, "attributes": attr}


def check_dataframe(df):
    """Convert input to polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pd.DataFrame):
        return pl.from_pandas(df.copy())
    else:
        return pl.DataFrame(df)


def check_series(s):
    """Convert input to polars Series."""
    if isinstance(s, pl.Series):
        return s
    elif isinstance(s, pd.Series):
        return pl.from_pandas(s)
    else:
        return pl.Series(s)
