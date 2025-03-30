__version__ = "1.6.0"

from . import basics, model
from .bins import *
from .example_data import *
from .notebook import *
from .props import *
from .stats import *
from .utils import *


# Create wrapper functions with deprecation errors
def _make_wrapper(func_name, module):
    def wrapper(*args, **kwargs):
        raise DeprecationWarning(
            f"{func_name}() is deprecated. Use {module}.{func_name}() instead."
        )
    return wrapper

# Model functions that need wrappers
model_functions = [
    "distr_plot",
    "make_train",
    "cross_validation",
    "regress",
    "rforest",
    "logistic",
    "mlp",
    "xgboost",
    "gains_plot"
]

# Basics classes that need wrappers
basics_functions = [
    "central_limit_theorem",
    "compare_means",
    "compare_props",
    "correlation",
    "cross_tabs",
    "goodness",
    "prob_calc",
    "single_mean",
    "single_prop"
]

# Create wrappers for model functions
for func_name in model_functions:
    globals()[func_name] = _make_wrapper(func_name, "model")

# Create wrappers for basics classes
for func_name in basics_functions:
    globals()[func_name] = _make_wrapper(func_name, "basics")

__all__ = ['model', 'basics'] + model_functions + basics_functions
