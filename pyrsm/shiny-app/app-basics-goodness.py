from shiny import App
import pyrsm as rsm
from pyrsm.radiant.goodness import *

data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
rc = basics_goodness(data_dct, descriptions_dct, open=True)
app = App(rc.shiny_ui(), rc.shiny_server)
