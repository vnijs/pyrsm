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

choices = {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"}


def ui_summary(self):
    return (
        ui.panel_conditional(
            "input.tabs == 'Summary'",
            ui.panel_well(
                ui.output_ui("ui_vars"),
                ui.input_select(
                    id="method",
                    label="Method:",
                    selected=self.state.get("method", "pearson"),
                    choices=choices,
                ),
                ui.input_numeric(
                    id="cutoff",
                    label="Correlation cutoff (abs):",
                    value=self.state.get("cutoff", 0),
                    min=0,
                    max=1,
                    step=0.05,
                ),
                ui.input_checkbox(
                    id="cov",
                    label="Show covariance matrix:",
                    value=self.state.get("cov", False),
                ),
                ui.input_numeric(
                    id="dec",
                    label="Decimals:",
                    value=self.state.get("cutoff", 2),
                    min=0,
                    step=1,
                ),
            ),
        ),
    )


def ui_plot(self):
    return ui.panel_well(
        ui.input_select(
            "nobs",
            "Number of data points plotted:",
            selected=self.state.get("nobs", None),
            choices={1000: "1,000", -1: "All"},
        ),
    )


def plots_extra(self):
    return (
        ui.input_select(
            "nobs",
            "Number of data points plotted:",
            selected=self.state.get("nobs", "1,000"),
            choices={1_000: "1,000", -1: "All"},
        ),
    )


class basics_correlation:
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
                "Basics > Correlation",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ui_summary(self),
                        # ui_plot(self),
                        ru.ui_plot(
                            self, {"corr": "Correlation plot"}, plots_extra(self)
                        ),
                    ),
                    ui.column(8, ru.ui_main_basics(self)),
                ),
            ),
            self.navbar,
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/basics/basics-correlation.ipynb",
                "Correlation example notebook",
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
        @output(id="ui_vars")
        @render.ui
        def ui_vars():
            isNum = get_data()["var_types"]["isNum"]
            return ui.input_select(
                id="vars",
                label="Variables:",
                selected=self.state.get("vars", []),
                choices=isNum,
                multiple=True,
                size=min(8, len(isNum)),
                selectize=False,
            )

        def estimation_code():
            data_name, code = (get_data()[k] for k in ["data_name", "code"])
            vars = rsm.ifelse(
                isinstance(input.vars(), tuple), list(input.vars()), input.vars()
            )
            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "vars": vars,
                "method": input.method(),
            }

            args_string = ru.drop_default_args(args, rsm.basics.correlation)
            return f"""rsm.basics.correlation({args_string})""", code

        show_code, estimate = mu.make_estimate(
            self,
            input,
            output,
            get_data,
            fun="basics.correlation",
            ret="cr",
            ec=estimation_code,
            run=False,
            debug=True,
        )

        def summary_code():
            args = {"cov": input.cov(), "cutoff": input.cutoff(), "dec": input.dec()}
            args_string = ru.drop_default_args(args, rsm.basics.correlation.summary)
            return f"""cr.summary({args_string})"""

        mu.make_summary(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="cr",
            sum_fun=rsm.basics.correlation.summary,
            sc=summary_code,
        )

        def plot_code():
            args = {"nobs": input.nobs()}
            args_string = ru.drop_default_args(
                args, rsm.basics.correlation.plot, ignore=["nobs"]
            )
            return f"""cr.plot({args_string})"""

        mu.make_plot(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="cr",
            pc=plot_code,
        )

        # --- section standard for all apps ---
        # stops returning code if moved to utils
        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)


def correlation(
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
        data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="salary")
    rc = basics_correlation(data_dct, descriptions_dct, state=state, code=code)
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
    correlation(debug=False)
