import os
from importlib import import_module
import pandas as pd


def load_data(pkg=None, name=None, dct=None):
    """
    Load example data included in the pyrsm package

    Parameters
    ----------
    pkg : str
        One of "data", "design", "basics", "model", or "multivariate"
        These string coincide with the dropdown menuss at https://radiant-rstats.github.io/docs/
        If None, datasets from all packages will be returned
    name : str
        Name of the dataset you want to load (e.g., "diamonds").
        If None, all datasets will be returned
    dct : dct
        Dictionary to add datasets to. For example, using globals() will add
        dataframes to the global environment in python. If None, a dictionary
        with datasets will be returned

    Examples
    --------

    import pyrsm as rsm
    rsm.load_data(pkg="basics", name="demand_uk", dct=globals())
    """

    base = "pyrsm.data"
    base_path = import_module(base).__path__[0]

    def load_data(file_path):
        return pd.read_parquet(file_path)

    def load_description(file_path):
        # read a text file into a string
        with open(f"{file_path}_description.md", "r") as file:
            descr = file.read()
        return descr

    def mkdct(spkg):
        data = {}
        description = {}
        for sp in spkg:
            package_path = os.path.join(base_path, sp)
            for file_name in os.listdir(package_path):
                if (name is None and not file_name.startswith(("__", "."))) or (
                    name is not None and file_name == f"{name}.parquet"
                ):
                    file_path = os.path.join(package_path, file_name)
                    key = file_name.replace(".parquet", "")
                    data[key] = load_data(file_path)
                    description[key] = load_description(file_path)
        return data, description

    if pkg is None:
        data, description = mkdct(
            [d for d in os.listdir(base_path) if not d.startswith("__")]
        )
    elif name is None:
        data, description = mkdct([pkg])
    elif pkg is not None and name is not None:
        data = load_data(os.path.join(base_path, pkg, f"{name}.parquet"))
        description = load_description(os.path.join(base_path, pkg, name))
    else:
        data = {}
        description = {}

    if dct is None:
        return data, description
    else:
        for key, val in data.items():
            dct[key] = val
        for key, val in description.items():
            dct[f"{key}_description"] = val
