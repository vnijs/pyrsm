from pathlib import Path
from shiny import App, ui
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from pyrsm.radiant import utils as ru
from pyrsm.radiant.compare_means import basics_compare_means
from pyrsm.radiant.cross_tabs import basics_cross_tabs
from pyrsm.radiant.data_view import data_view
from pyrsm.radiant.goodness import basics_goodness
from pyrsm.radiant.logistic import model_logistic
from pyrsm.radiant.probability_calculator import basics_probability_calculator
from pyrsm.radiant.regress import model_regress
from pyrsm.radiant.single_mean import basics_single_mean

# import polars as pl
# import pandas as pd

# linear regression app
# polars reads dates from parquet just fine, pandas does not
# date format is maintained when using to_pandas
# data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="diamonds", polars=True)
# data_dct["diamonds"] = data_dct["diamonds"].to_pandas()
data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="diamonds")
data_dct.update({"diamonds100": data_dct["diamonds"].sample(100)})

# main path to www folder
www_dir = Path(__file__).parent.parent / "radiant" / "www"

# serve static files
app_static = StaticFiles(directory=www_dir, html=False)

# data > view
data_dct, descriptions_dct = ru.get_dfs(pkg="data")
rc = data_view(data_dct, descriptions_dct, code=True)
app_data = App(rc.shiny_ui(ru.radiant_navbar()), rc.shiny_server, debug=False)

# probability calculator mean app
rc = basics_probability_calculator(code=True)
app_pc = App(rc.shiny_ui(ru.radiant_navbar()), rc.shiny_server, debug=False)

# single mean app
data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="demand_uk")
rc = basics_single_mean(data_dct, descriptions_dct, code=True)
app_sm = App(rc.shiny_ui(ru.radiant_navbar()), rc.shiny_server, debug=False)

# compare means app
data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="salary")
rc = basics_compare_means(data_dct, descriptions_dct, code=True)
app_cm = App(rc.shiny_ui(ru.radiant_navbar()), rc.shiny_server, debug=False)

# cross-tabs app
data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
rc = basics_cross_tabs(data_dct, descriptions_dct, code=True)
app_ct = App(rc.shiny_ui(ru.radiant_navbar()), rc.shiny_server, debug=False)

# goodness app
data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
rc = basics_goodness(data_dct, descriptions_dct, code=True)
app_gf = App(rc.shiny_ui(ru.radiant_navbar()), rc.shiny_server, debug=False)

# linear regression app
data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="diamonds")
data_dct.update({"diamonds100": data_dct["diamonds"].sample(100)})

descriptions_dct.update({"diamonds100": descriptions_dct["diamonds"]})
rc = model_regress(data_dct, descriptions_dct, code=True)
app_regress = App(rc.shiny_ui(ru.radiant_navbar()), rc.shiny_server)

# logistic regression app
data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="titanic")
rc = model_logistic(data_dct, descriptions_dct, code=True)
app_logistic = App(rc.shiny_ui(ru.radiant_navbar()), rc.shiny_server)

ui_nav = ui.page_navbar(
    ru.head_content(),
    ru.radiant_navbar(),
    ru.ui_stop(),
    title="Radiant for Python",
    inverse=True,
    id="navbar_id",
)
app_home = App(ui_nav, None)

# ---- combine apps ----
routes = [
    Mount("/basics/prob-calc/", app=app_pc),
    Mount("/basics/single-mean/", app=app_sm),
    Mount("/basics/compare-means/", app=app_cm),
    Mount("/basics/goodness/", app=app_gf),
    Mount("/basics/cross-tabs/", app=app_ct),
    Mount("/models/regress", app=app_regress),
    Mount("/models/logistic", app=app_logistic),
    Mount("/www", app=app_static),
    Mount("/", app=app_data),  # MUST BE LAST!!!!
]

app = Starlette(debug=True, routes=routes)

# some combination of https://github.com/rstudio/py-shiny/issues/482
# and multiple inheritance might work to have a shared data object
# https://chat.openai.com/share/7dca0c72-bf2d-49c4-8ce7-f7dfb88b10b8
