from shiny import App
import pyrsm as rsm
from pyrsm.radiant.single_mean import *

data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="demand_uk")
rc = basics_single_mean(data_dct, descriptions_dct, open=True)
app = App(rc.shiny_ui(), rc.shiny_server, debug=False)
