from shiny import App
import pyrsm as rsm
from pyrsm.radiant.goodness_of_fit import *

newspaper, newspaper_description = rsm.load_data(pkg="basics", name="newspaper")
rc = basics_goodness_of_fit(
    {"newspaper": newspaper}, {"newspaper": newspaper_description}, open=True
)
app = App(rc.shiny_ui(), rc.shiny_server)
