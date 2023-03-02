import pickle


def save_state(dct, path):
    """
    Store the (partial) state of a Jupyter notebook

    Parameters
    ----------
    dct : dict
        Dictionary of python objects to store (e.g., globals())
    path : str
        File path to use to store state. Use .state.pkl as
        the file name extension

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
    try:
        with open(path, "wb") as f:
            pickle.dump(state, f)
    except Exception as err:
        print(err)
        print("\nSome objects in globals() could not be 'pickled'")
        print("Either remove those objects or explicitly pass in")
        print("a dictionary with the objects you want to store")


def load_state(path, dct=None):
    """
    Re-store the (partial) state of a Jupyter notebook

    Parameters
    ----------
    path : str
        Path to state file location
    dct : dict
        Dictionary to add python objects to (e.g., globals()).
        If None, a dictionary of objects will be returned

    Examples
    --------
    load_state("my-notebook.state.pkl", globals())
    """

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
