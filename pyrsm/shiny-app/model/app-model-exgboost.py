from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from pathlib import Path
from shiny import App
from pyrsm.radiant.model.exgboost import model_exgboost
import pyrsm.radiant.utils as ru

www_dir = Path(__file__).parent.parent.parent / "radiant" / "www"
app_static = StaticFiles(directory=www_dir, html=False)

data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="titanic")
rc = model_exgboost(data_dct, descriptions_dct, state=None, code=True)

routes = [
    Mount("/www", app=app_static),
    Mount("/", app=App(rc.shiny_ui, rc.shiny_server, debug=False)),
]
app = Starlette(debug=True, routes=routes)
