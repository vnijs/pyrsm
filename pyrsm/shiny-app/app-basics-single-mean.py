from shiny import App
import pyrsm as rsm
from pyrsm.radiant.single_mean import *

demand_uk, demand_uk_description = rsm.load_data(pkg="basics", name="demand_uk")
rc = basics_single_mean(
    {"demand_uk": demand_uk}, {"demand_uk": demand_uk_description}, open=True
)
app = App(rc.shiny_ui(), rc.shiny_server, debug=False)
