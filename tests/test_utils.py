import pandas as pd
from pyrsm.utils import add_description, ifelse


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

