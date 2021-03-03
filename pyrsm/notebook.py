import pickle
import ipynbname


def save_state(dct, path=None):
    """
    Store the (partial) state of a Jupyter notebook

    Parameters
    ----------
    dct : dict
        Dictionary of python objects to store (e.g., globals())
    path : str
        File path to use to store state. If none, (try) to use the notebook name
        If the notebook name cannot be determined, uses 'notebook.state.pkl'

    Examples
    --------
    save_state(globals(), path="my-notebook.state.pkl")
    """

    remove_keys = [
        "In",
        "Out",
        "get_ipython",
        "exit",
        "quit",
        "json",
        "sys",
        "NamespaceMagics",
        "state",
        "remove_keys",
        "remove_types",
    ]
    remove_types = [
        "<class 'module'>",
        "<class 'function'>",
        "<class 'builtin_function_or_method'>",
        "<class 'abc.ABCMeta'>",
        "<class 'type'>",
        "<class '_io.BufferedReader'>",
        "<class 'weakref'>",
    ]
    state = {
        key: val
        for key, val in dct.items()
        if (
            not key.startswith("_")
            and (key not in remove_keys)
            and (str(type(val)) not in remove_types)
        )
    }
    if path is None:
        try:
            path = ipynbname.name() + ".state.pkl"
        except Exception as err:
            print(err)
            print("\nCould not determine the notebook name")
            print("Using 'notebook.state.pkl' instead")
            path = "notebook.state.pkl"

    try:
        with open(path, "wb") as f:
            pickle.dump(state, f)
    except Exception as err:
        print(err)
        print("\nSome objects in globals() could not be 'pickled'")
        print("Either remove those objects or explicitly pass in")
        print("a dictionary with the objects you want to store")


def load_state(dct=None, path=None):
    """
    Re-store the (partial) state of a Jupyter notebook

    Parameters
    ----------
    dct : dict
        Dictionary to add python objects to (e.g., globals()).
        If None, a dictionary with objects will be returned
    path : str
        Path to use to store state. If none, (try) to use the current notebook name.
        If the notebook name cannot be determined, uses 'notebook.state.pkl'

    Examples
    --------
    load_state(globals(), path="my-notebook.state.pkl")
    """
    if path is None:
        try:
            path = ipynbname.name() + ".state.pkl"
        except Exception as err:
            print(err)
            print("\nCould not determine the notebook name")
            print("Loading 'notebook.state.pkl' if available")
            path = "notebook.state.pkl"

    try:
        with open(path, "rb") as f:
            g = pickle.load(f)

        if dct is None:
            return g
        else:
            # using the mutability feature in python
            for key, val in g.items():
                dct[key] = val
    except Exception as err:
        print(err)
        print("\nCould not load file: " + path)
