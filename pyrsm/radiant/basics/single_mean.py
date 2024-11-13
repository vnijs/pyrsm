from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request as StarletteRequest
from shiny import App, render, ui, reactive, Inputs, Outputs, Session, req
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
import pyrsm.radiant.model.model_utils as mu


def ui_summary(self):
    return (
        ui.panel_conditional(
            "input.tabs == 'Summary'",
            ui.panel_well(
                ui.output_ui("ui_var"),
                ui.input_select(
                    id="alt_hyp",
                    label="Alternative hypothesis:",
                    selected=self.state.get("alt_hyp", "two-sided"),
                    choices={
                        "two-sided": "Two sided",
                        "greater": "Greater than",
                        "less": "Less than",
                    },
                ),
                ui.input_slider(
                    id="conf",
                    label="Confidence level:",
                    min=0,
                    max=1,
                    value=self.state.get("conf", 0.95),
                ),
                ui.input_numeric(
                    id="comp_value",
                    label="Comparison value:",
                    value=self.state.get("comp_value", 0),
                ),
            ),
        ),
    )


choices = {
    # "None": "None",
    "hist": "Histogram",
    # "sim": "Simulate",
}


class basics_single_mean:
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
            ui.nav_panel(
                "Basics > Single mean",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ui_summary(self),
                        ru.ui_plot(self, choices),
                    ),
                    ui.column(8, ru.ui_main_basics()),
                ),
            ),
            self.navbar,
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/basics/basics-single-mean.ipynb",
                "Single mean example notebook",
            ),
            ru.ui_stop(),
            title="Radiant for Python",
            inverse=False,
            id="navbar_id",
        )

    def shiny_server(self, input: Inputs, output: Outputs, session: Session):
        # --- section standard for all apps ---
        get_data = ru.make_data_elements(self, input, output, session)

        def update_state():
            with reactive.isolate():
                ru.dct_update(self, input)

        session.on_ended(update_state)

        # --- section unique to each app ---
        @output(id="ui_var")
        @render.ui
        def ui_var():
            isNum = get_data()["var_types"]["isNum"]
            return ui.input_select(
                id="var",
                label="Variable (select one)",
                selected=self.state.get("var", None),
                choices=isNum,
            )

        def estimation_code():
            data_name, code = (get_data()[k] for k in ["data_name", "code"])
            req(input.comp_value() is not None, "comp_value must be specified")

            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "var": input.var(),
                "alt_hyp": input.alt_hyp(),
                "conf": input.conf(),
                "comp_value": input.comp_value(),
            }

            args_string = ru.drop_default_args(args, rsm.basics.single_mean)
            return f"""rsm.basics.single_mean({args_string})""", code

        show_code, estimate = mu.make_estimate(
            self,
            input,
            output,
            get_data,
            fun="basics.single_mean",
            ret="sm",
            ec=estimation_code,
            run=False,
            debug=True,
        )

        def summary_code():
            return """sm.summary()"""

        mu.make_summary(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="sm",
            sum_fun=rsm.basics.single_mean.summary,
            sc=summary_code,
        )

        mu.make_plot(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="sm",
        )

        # --- section standard for all apps ---
        # stops returning code if moved to utils
        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)


def single_mean(
    data_dct: dict = None,
    descriptions_dct: dict = None,
    state: dict = None,
    code: bool = True,
    host: str = "",
    port: int = 8000,
    log_level: str = "warning",
    debug: bool = False,
):
    """
    Launch a Radiant-for-Python app for single_mean hypothesis testing
    """
    host = ru.set_host(host)
    if data_dct is None:
        data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="demand_uk")
    rc = basics_single_mean(data_dct, descriptions_dct, state=state, code=code)
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
    www_dir = Path(__file__).parent.parent.parent / "radiant" / "www"
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
    single_mean(debug=False)
