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
        dataframes to the global environment in python. If None, a dictioary
        with datasets will be returned

    Examples
    --------
    load_data(globals(), path="my-notebook.state.pkl")
    """

    base = "pyrsm.data"
    base_path = import_module(base).__path__[0]

    def load(file_path):
        return pd.read_pickle(file_path)

    def mkdct(spkg):
        data = {}
        for sp in spkg:
            package_path = os.path.join(base_path, sp)
            for file_name in os.listdir(package_path):
                if (name is None and not file_name.startswith(("__", "."))) or (
                    name is not None and file_name == f"{name}.pkl"
                ):
                    file_path = os.path.join(package_path, file_name)
                    key = file_name.replace(".pkl", "")
                    data[key] = load(file_path)
        return data

    if pkg is None:
        data = mkdct([d for d in os.listdir(base_path) if not d.startswith("__")])
    elif name is None:
        data = mkdct([pkg])
    elif pkg is not None and name is not None:
        data = {name: load(os.path.join(base_path, pkg, f"{name}.pkl"))}
    else:
        data = {}

    if dct is None:
        return data
    else:
        for key, val in data.items():
            dct[key] = val
