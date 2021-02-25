import pickle
import ipynbname


def save_state(path=None):
    store = {}
    globals_keys = dict(globals())
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

    for key, val in globals_keys.items():
        if (
            not key.startswith("_")
            and (key not in remove_keys)
            and (str(type(val)) not in remove_types)
        ):
            store[key] = val

    if path is None:
        path = ipynbname.name() + ".state.pkl"

    with open(path, "wb") as f:
        pickle.dump(store, f)


def load_state(path=None):
    if path is None:
        path = ipynbname.name() + ".state.pkl"

    with open(path, "rb") as f:
        gkey = pickle.load(f)

    for key, val in gkey.items():
        globals()[key] = val
