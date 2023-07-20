from shiny import App
import pyrsm as rsm
from pyrsm.radiant.compare_means import *

data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="salary")
rc = basics_compare_means(data_dct, descriptions_dct, code=True)
app = App(rc.shiny_ui(), rc.shiny_server, debug=False)
