from pathlib import Path
from shiny import App
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from pyrsm.radiant import utils as ru
from pyrsm.radiant.data.data_view import data_view
from pyrsm.radiant.basics.probability_calculator import basics_probability_calculator
from pyrsm.radiant.basics.single_mean import basics_single_mean
from pyrsm.radiant.basics.single_prop import basics_single_prop
from pyrsm.radiant.basics.compare_means import basics_compare_means
from pyrsm.radiant.basics.compare_props import basics_compare_props
from pyrsm.radiant.basics.cross_tabs import basics_cross_tabs
from pyrsm.radiant.basics.goodness import basics_goodness
from pyrsm.radiant.basics.correlation import basics_correlation
from pyrsm.radiant.model.regress import model_regress
from pyrsm.radiant.model.logistic import model_logistic
import nest_asyncio
import uvicorn
import webbrowser
import sys
import tempfile


def radiant(
    data_dct: dict = None,
    descriptions_dct: dict = None,
    state: dict = None,
    code: bool = True,
    # host: str = "0.0.0.0",
    host: str = "localhost",
    port: int = 8000,
    log_level: str = "warning",
    debug: bool = False,
):
    """
    Launch the Radiant-for-Python app
    """

    # redirect stdout and stderr to the temporary file
    if not debug:
        org_stdout = sys.stdout
        org_stderr = sys.stderr
        temp = tempfile.NamedTemporaryFile()
        temp_file = open(temp.name, "w")
        sys.stdout = temp_file
        sys.stderr = temp_file

    data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="diamonds")
    data_dct.update({"diamonds100": data_dct["diamonds"].sample(100)})

    # main path to www folder
    www_dir = Path(__file__).parent.parent / "radiant" / "www"

    # serve static files
    app_static = StaticFiles(directory=www_dir, html=False)

    # data > view
    data_dct, descriptions_dct = ru.get_dfs(pkg="data")
    rc = data_view(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_data = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # probability calculator mean app
    rc = basics_probability_calculator(
        state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_pc = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # single mean app
    data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="demand_uk")
    rc = basics_single_mean(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_sm = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # compare means app
    data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="salary")
    rc = basics_compare_means(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_cm = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # single prop
    data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="consider")
    rc = basics_single_prop(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_sp = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # compare props
    data_dct, descriptions_dct = ru.get_dfs(pkg="data", name="titanic")
    rc = basics_compare_props(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_cp = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # cross-tabs app
    data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
    rc = basics_cross_tabs(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_ct = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # goodness app
    data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
    rc = basics_goodness(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_gf = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # correlation app
    data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="salary")
    rc = basics_correlation(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_cr = App(rc.shiny_ui, rc.shiny_server, debug=False)

    # linear regression app
    data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="diamonds")
    data_dct.update({"diamonds100": data_dct["diamonds"].sample(100)})

    descriptions_dct.update({"diamonds100": descriptions_dct["diamonds"]})
    rc = model_regress(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_regress = App(rc.shiny_ui, rc.shiny_server)

    # logistic regression app
    data_dct, descriptions_dct = ru.get_dfs(pkg="model", name="titanic")
    rc = model_logistic(
        data_dct, descriptions_dct, state=None, code=True, navbar=ru.radiant_navbar()
    )
    app_logistic = App(rc.shiny_ui, rc.shiny_server)

    # ---- combine apps ----
    routes = [
        Mount("/basics/prob-calc/", app=app_pc),
        Mount("/basics/single-mean/", app=app_sm),
        Mount("/basics/compare-means/", app=app_cm),
        Mount("/basics/single-prop/", app=app_sp),
        Mount("/basics/compare-props/", app=app_cp),
        Mount("/basics/cross-tabs/", app=app_ct),
        Mount("/basics/goodness/", app=app_gf),
        Mount("/basics/correlation/", app=app_cr),
        Mount("/models/regress/", app=app_regress),
        Mount("/models/logistic/", app=app_logistic),
        Mount("/www/", app=app_static),
        Mount("/", app=app_data),  # MUST BE LAST!!!!
    ]

    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    print(f"Listening on http://{host}:{port}")
    ru.message()

    uvicorn.run(
        Starlette(debug=debug, routes=routes),
        host=host,
        port=port,
        log_level=log_level,
    )

    if not debug:
        sys.stdout = org_stdout
        sys.stderr = org_stderr
        temp_file.close()


if __name__ == "__main__":
    radiant(debug=False)
