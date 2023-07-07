from shiny import App
import pyrsm as rsm

from pyrsm.radiant.regress import *

diamonds, diamonds_description = rsm.load_data(pkg="data", name="diamonds")
rc = model_regress(
    {"diamonds": diamonds, "diamonds100": diamonds.sample(100)},
    {"diamonds": diamonds_description, "diamonds100": diamonds_description},
    open=True,
)
app = App(rc.shiny_ui(), rc.shiny_server)
