from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request as StarletteRequest
from shiny import App, ui, reactive, Inputs, Outputs, Session
from pathlib import Path
import webbrowser
import nest_asyncio
import uvicorn
import signal
import os
import sys
import tempfile
import pyrsm as rsm
import pyrsm.radiant.utils as ru

# make a radiant_data class with a get_data method
# this can be called from the sub-classes
# https://chat.openai.com/share/0c566cff-6f02-4e55-aec4-e6b379a9aadf
# within each sub-class that method will be made into a reactive.Calc
# check if filters set in the data (?) or radiant_data are passed to the
# sub-classes because dictionaries are mutable ?
# also is the "super" class (data / radiant_data) the place to store state
# from all sub-classes?
# if this works, how to best "rip-out" the data parts of each sub-app and still be
# able to use them individually with a data-filter tab?

# on-session-ended should be used to store information from "input" into the state
# dictionary

# possible to only activate the apps you use? the below suggests you can
# add routes dynamically so ... maybe? ask @joe
# https://github.com/rstudio/py-shiny/issues/352

# see here for how you might be able to use the state dictionary for an
# individual user on a server
# https://discord.com/channels/1109483223987277844/1129448698611511326/1129732244353851432


class data_view:
    def __init__(
        self, datasets: dict, descriptions=None, state=None, code=True, navbar=None
    ) -> None:
        ru.init(
            self,
            datasets,
            descriptions=descriptions,
            state=state,
            code=code,
            navbar=navbar,
        )

    def shiny_ui(self, request: StarletteRequest):
        return ui.page_navbar(
            ru.head_content(),
            ui.nav(
                "Data > View",
                ui.row(
                    ui.column(3, ru.ui_view(self)),
                    ui.column(
                        8,
                        ui.navset_tab_card(
                            ui.nav("View", ru.ui_data_main()),
                        ),
                    ),
                ),
            ),
            self.navbar,
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/data-view.ipynb",
                "Data > View example notebook",
            ),
            ru.ui_stop(),
            title="Radiant for Python",
            inverse=True,
            id="navbar_id",
        )

    def shiny_server(self, input: Inputs, output: Outputs, session: Session):
        # --- section standard for all apps ---
        ru.make_data_elements(self, input, output, session)

        def update_state():
            with reactive.isolate():
                ru.dct_update(self, input)

        session.on_ended(update_state)

        # --- section standard for all apps ---
        # stops returning code if moved to utils
        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)


def view(
    data_dct: dict = None,
    descriptions_dct: dict = None,
    state: dict = None,
    code: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
    debug: bool = False,
):
    """
    Launch a Radiant-for-Python app for single_mean hypothesis testing
    """
    if data_dct is None:
        data_dct, descriptions_dct = ru.get_dfs(pkg="data")
    rc = data_view(data_dct, descriptions_dct, state=state, code=code)
    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    print(f"Listening on http://{host}:{port}")
    ru.message()

    # redirect stdout and stderr to the temporary file
    if not debug:
        org_stdout = sys.stdout
        org_stderr = sys.stderr
        temp = tempfile.NamedTemporaryFile()
        temp_file = open(temp.name, "w")
        sys.stdout = temp_file
        sys.stderr = temp_file

    app = App(rc.shiny_ui, rc.shiny_server)
    www_dir = Path(__file__).parent.parent / "radiant" / "www"
    app_static = StaticFiles(directory=www_dir, html=False)

    routes = [
        Mount("/www", app=app_static),
        Mount("/", app=app),
    ]

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
    # demand_uk, demand_uk_description = rsm.load_data(pkg="basics", name="demand_uk")
    # data_dct, descriptions_dct = ru.get_dfs(name="demand_uk")
    # single_mean(data_dct, descriptions_dct, code=True)
    view(debug=False)
