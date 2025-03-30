from pathlib import Path

from shiny import App
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

import pyrsm.radiant.utils as ru
from pyrsm.radiant.model.regress import model_regress

www_dir = Path(__file__).parent.parent.parent / "radiant" / "www"
app_static = StaticFiles(directory=www_dir, html=False)

# data_dct, descriptions_dct = ru.get_dfs(pkg="model")
data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="diamonds")
data_dct.update({"diamonds100": data_dct["diamonds"].sample(100)})
descriptions_dct.update({"diamonds100": descriptions_dct["diamonds"]})
rc = model_regress(data_dct, descriptions_dct, state=None, code=True)

routes = [
    Mount("/www", app=app_static),
    Mount("/", app=App(rc.shiny_ui, rc.shiny_server, debug=False)),
]
app = Starlette(debug=True, routes=routes)
