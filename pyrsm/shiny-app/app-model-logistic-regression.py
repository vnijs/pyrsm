from shiny import App
import pyrsm as rsm

from pyrsm.radiant.logistic import *

titanic, titanic_description = rsm.load_data(pkg="data", name="titanic")
rc = model_logistic({"titanic": titanic}, {"titanic": titanic_description}, open=True)
app = App(rc.shiny_ui(), rc.shiny_server)
