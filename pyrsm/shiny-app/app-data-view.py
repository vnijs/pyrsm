from pathlib import Path

from shiny import App
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

import pyrsm.radiant.utils as ru
from pyrsm.radiant.data.data_view import data_view

www_dir = Path(__file__).parent.parent / "radiant" / "www"
app_static = StaticFiles(directory=www_dir, html=False)

data_dct, descriptions_dct = ru.get_dfs(pkg="data")
rc = data_view(data_dct, descriptions_dct, state=None, code=True)

routes = [
    Mount("/www", app=app_static),
    Mount("/", app=App(rc.shiny_ui, rc.shiny_server, debug=False)),
]
app = Starlette(debug=True, routes=routes)
