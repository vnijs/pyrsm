from shiny import App
from pyrsm.radiant.logistic import *

data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="titanic")
rc = model_logistic(data_dct, descriptions_dct, code=True)
app = App(rc.shiny_ui(), rc.shiny_server)
