import numpy as np


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
