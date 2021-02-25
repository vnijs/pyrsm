import pickle
import ipynbname


def save_state(path=None):
    remove_keys = {
        "In",
        "Out",
        "get_ipython",
        "exit",
        "quit",
        "json",
        "sys",
        "NamespaceMagics",
        "store",
        "remove_keys",
        "remove_types",
    }
    remove_types = {
        "<class 'module'>",
        "<class 'function'>",
        "<class 'builtin_function_or_method'>",
        "<class 'abc.ABCMeta'>",
        "<class 'type'>",
        "<class '_io.BufferedReader'>",
    }

    state = {
        key: val
        for key, val in globals().items()
        if (
            not key.startswith("_")
            and (key not in remove_keys)
            and (str(type(val)) not in remove_types)
        )
    }

    if path is None:
        path = ipynbname.name() + ".state.pkl"

    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path=None):
    if path is None:
        path = ipynbname.name() + ".state.pkl"

    with open(path, "rb") as f:
        g = pickle.load(f)

    for key, val in g.items():
        globals()[key] = val
