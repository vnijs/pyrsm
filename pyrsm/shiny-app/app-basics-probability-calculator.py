from shiny import App
import pyrsm as rsm
from pyrsm.radiant.probability_calculator import *

rc = basics_probability_calculator(code=True)
app = App(rc.shiny_ui(), rc.shiny_server, debug=False)
