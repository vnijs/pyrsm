from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request as StarletteRequest
from shiny import App, ui, render, reactive, Inputs, Outputs, Session
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
import pyrsm.radiant.model_utils as mu

choices = {
    "None": "None",
    # "dist": "Distribution",
    # "corr": "Correlation",
    "pred": "Prediction plots",
    "vimp": "Permutation importance",
    "or": "OR plot",
}

controls = {
    "ci": "Confidence intervals",
    "vif": "VIF",
}


def summary_extra(self):
    return (
        ui.input_radio_buttons(
            "show_interactions",
            "Interactions:",
            selected=self.state.get("show_interactions", None),
            choices={0: "None", 2: "2-way", 3: "3-way"},
            inline=True,
        ),
        ui.panel_conditional(
            "input.show_interactions > 0",
            ui.output_ui("ui_interactions"),
        ),
        ui.input_checkbox_group(
            "controls",
            "Additional output:",
            selected=self.state.get("controls", None),
            choices=controls,
        ),
        ui.output_ui("ui_evar_test"),
    )


def plots_extra(self):
    return (
        ui.panel_conditional(
            "input.plots == 'corr'",
            ui.input_select(
                "nobs",
                "Number of data points plotted:",
                selected=self.state.get("nobs", None),
                choices={1000: "1,000", -1: "All"},
            ),
        ),
        ui.panel_conditional(
            "input.plots == 'pred'",
            ui.output_ui("ui_incl_evar"),
            ui.output_ui("ui_incl_interactions"),
        ),
    )


class model_logistic:
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
                "Model > Logistic regression (GLM)",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ru.ui_summary(summary_extra(self)),
                        mu.ui_predict(self),
                        ru.ui_plot(self, choices, plots_extra(self)),
                    ),
                    ui.column(8, ru.ui_main_model()),
                ),
            ),
            self.navbar,
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/model-logistic-regression.ipynb",
                "Logistic regression (GLM) example notebook",
            ),
            ru.ui_stop(),
            title="Radiant for Python",
            inverse=True,
            id="navbar_id",
        )

    def shiny_server(self, input: Inputs, output: Outputs, session: Session):
        # --- section standard for all apps ---
        get_data = ru.make_data_elements(self, input, output, session)

        def update_state():
            with reactive.isolate():
                ru.dct_update(self, input)

        session.on_ended(update_state)

        # --- section standard for all model apps ---
        ru.reestimate(input)

        # --- section unique to each app ---
        mu.make_model_inputs(self, input, output, get_data, "isBin")

        @output(id="ui_lev")
        @render.ui
        def ui_lev():
            levs = list(get_data()["data"][input.rvar()].unique())
            return ui.input_select(
                id="lev",
                label="Choose level:",
                selected=self.state.get("lev", None),
                choices=levs,
            )

        mu.make_int_inputs(self, input, output, get_data)
        show_code, estimate = mu.make_estimate(
            self, input, output, get_data, fun="logistic", ret="lr", debug=False
        )
        mu.make_summary(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="lr",
            sum_fun=rsm.logistic.summary,
        )
        mu.make_predict(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="lr",
            pred_fun=rsm.logistic.predict,
        )
        mu.make_plot(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="lr",
        )

        # --- section standard for all apps ---
        # stops returning code if moved to utils
        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)


def logistic(
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
    Launch a Radiant-for-Python app for logistic regression analysis
    """
    if data_dct is None:
        data_dct, descriptions_dct = ru.get_dfs(pkg="model")
    rc = model_logistic(data_dct, descriptions_dct, state=state, code=code)
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
    # titanic, titanic_description = rsm.load_data(pkg="data", name="titanic")
    # logistic({"titanic": titanic}, {"titanic": titanic_description}, code=True)
    logistic(debug=False)
