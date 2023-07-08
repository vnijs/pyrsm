from shiny import App
import pyrsm as rsm
from pyrsm.radiant.cross_tabs import *

newspaper, newspaper_description = rsm.load_data(pkg="basics", name="newspaper")
rc = basics_cross_tabs(
    {"newspaper": newspaper}, {"newspaper": newspaper_description}, open=True
)
app = App(rc.shiny_ui(), rc.shiny_server)
