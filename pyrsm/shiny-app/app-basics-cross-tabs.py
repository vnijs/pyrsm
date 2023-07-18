from shiny import App
import pyrsm as rsm
from pyrsm.radiant.cross_tabs import *

data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
rc = basics_cross_tabs(data_dct, descriptions_dct, open=True)
app = App(rc.shiny_ui(), rc.shiny_server)
