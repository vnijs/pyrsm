from pyrsm import add_description
import pandas as pd


md = """# Data Description

The variables in the dataset are a and b

* a: The first variable
* b: The second variable"""


def test_add_description():
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df = add_description(df, md)
    assert df.description == md, "Description not attached"
