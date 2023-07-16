from shiny import App
from pyrsm.radiant.logistic import *

data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="titantic")
rc = model_logistic(data_dct, descriptions_dct, open=True)
app = App(rc.shiny_ui(), rc.shiny_server)
