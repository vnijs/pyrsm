import pandas as pd
import numpy as np
from pyrsm.utils import (
    add_description,
    ifelse,
    levels_list,
    expand_grid,
    table2data,
    setdiff,
    union,
    intersect,
)


md = """# Data Description

The variables in the dataset are a and b

* a: The first variable
* b: The second variable"""


def test_add_description():
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df = add_description(df, md)
    assert df.description == md, "Description not attached"


def test_ifelse_true():
    assert (
        ifelse(3 > 2, "greater", "smaller") == "greater"
    ), "Logical comparison in ifelse incorrect"


def test_ifelse_false():
    assert (
        ifelse(2 > 3, "greater", "smaller") == "smaller"
    ), "Logical comparison in ifelse incorrect"


def test_ifelse_array():
    assert all(
        ifelse(np.array([2, 3, 4]) > 2, 1, 0) == np.array([0, 1, 1])
    ), "Logical comparison of np.array in ifelse incorrect"


df = pd.DataFrame({"var1": ["a", "b", "a"], "var2": [1, 2, 1]})
dct = {"var1": ["a", "b"], "var2": [1, 2]}


def test_level_list():
    assert levels_list(df) == dct, "Levels list created incorrect dictionary"


def test_expand_grid():
    edf = expand_grid(dct)
    assert list(edf.loc[1].values) == ["a", 2], "Expand grid row 1 incorrect"
    assert list(edf.loc[2].values) == ["b", 1], "Expand grid row 3 incorrect"


def test_table2data():
    t2d = table2data(df.assign(freq=[3, 4, 5]), "freq")
    assert t2d.size == 36, "Number of rows from table2data is incorrect"
    assert (
        t2d["var1"] == "a"
    ).sum() == 8, "Number of 'a' values incorrect in table2data"


def test_setdiff():
    assert setdiff(["a", "b", "c"], ["b", "x"]) == [
        "a",
        "c",
    ], "Set difference incorrect"


def test_union():
    assert union(["a", "b", "c"], ["b", "x"]) == ["a", "b", "c", "x"], "Union incorrect"


def test_intersect():
    assert intersect(["a", "b", "c"], ["b", "x"]) == ["b"], "Union incorrect"
