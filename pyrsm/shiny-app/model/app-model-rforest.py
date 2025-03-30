from pathlib import Path

from shiny import App
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

import pyrsm.radiant.utils as ru
from pyrsm.radiant.model.rforest import model_rforest

www_dir = Path(__file__).parent.parent.parent / "radiant" / "www"
app_static = StaticFiles(directory=www_dir, html=False)

data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="titanic")
rc = model_rforest(data_dct, descriptions_dct, state=None, code=True)

routes = [
    Mount("/www", app=app_static),
    Mount("/", app=App(rc.shiny_ui, rc.shiny_server, debug=False)),
]
app = Starlette(debug=True, routes=routes)

# Prediction test for Data & Command option
# import pyrsm as rsm
# data_dct, descriptions_dct = rsm.radiant.utils.get_dfs(pkg="model", name="titanic")
# lr = rsm.model.logistic(
#   data=data_dct, rvar="price", evar=["carat", "clarity", "cut"]
# )
# reg.predict(data=diamonds, data_cmd={"carat": 1}, ci=True)
# reg.predict(data=diamonds, cmd={"carat": [1]}, ci=True)
