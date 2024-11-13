from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from pathlib import Path
from shiny import App
from pyrsm.radiant.basics.compare_means import basics_compare_means
import pyrsm.radiant.utils as ru

www_dir = Path(__file__).parent.parent.parent / "radiant" / "www"
app_static = StaticFiles(directory=www_dir, html=False)

data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="salary")
# data_dct, descriptions_dct = ru.get_dfs(pkg="data", name="diamonds")
rc = basics_compare_means(data_dct, descriptions_dct, state=None, code=True)

routes = [
    Mount("/www", app=app_static),
    Mount("/", app=App(rc.shiny_ui, rc.shiny_server, debug=False)),
]
app = Starlette(debug=True, routes=routes)

# from shiny import ui

# app_ui = ui.page_fluid(
#     ui.navset_tab(
#         ui.nav_panel("Tab 1", ui.h2("Content for Tab 1")),
#         ui.nav_panel("Tab 2", ui.h2("Content for Tab 2")),
#     )
# )
