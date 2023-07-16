from shiny import App
from pyrsm.radiant.regress import *

# data_dct, descriptions_dct = ru.get_dfs(pkg="model")
data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="diamonds")
rc = model_regress(data_dct, descriptions_dct, open=True)
app = App(rc.shiny_ui(), rc.shiny_server)
