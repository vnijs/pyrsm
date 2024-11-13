from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request as StarletteRequest
from pathlib import Path
from shiny import App, render, ui, reactive, Inputs, Outputs, Session, req
import webbrowser
import nest_asyncio
import uvicorn
import signal
import os
import sys
import tempfile
import pyrsm as rsm
from pyrsm.utils import ifelse
import pyrsm.radiant.utils as ru
import pyrsm.radiant.model.model_utils as mu


def ui_summary(self):
    return ui.panel_conditional(
        "input.tabs == 'Summary'",
        ui.panel_well(
            ui.output_ui("ui_var1"),
            ui.output_ui("ui_var2"),
            ui.output_ui("ui_lev"),
            ui.output_ui("ui_combos"),
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
            ui.input_checkbox(
                id="extra",
                label="Show extra statistics:",
                value=self.state.get("extra", False),
            ),
            ui.panel_conditional(
                "input.extra == true",
                ui.input_slider(
                    id="conf",
                    label="Confidence level:",
                    min=0,
                    max=1,
                    value=self.state.get("conf", 0.95),
                ),
            ),
            ui.input_radio_buttons(
                id="adjust",
                label="Multiple comp. adjustment:",
                selected=self.state.get("adjust", None),
                choices={
                    "None": "None",
                    "bonferroni": "Bonferroni",
                },
                inline=True,
            ),
        ),
    )


choices = {
    "None": "None",
    "bar": "Bar chart",
    "dodge": "Dodge plot",
}


class basics_compare_props:
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
                "Basics > Compare props",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ui_summary(self),
                        ru.ui_plot(self, choices, None),
                    ),
                    ui.column(8, ru.ui_main_basics(self)),
                ),
            ),
            self.navbar,
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/basics/basics-compare-props.ipynb",
                "Compare props example notebook",
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
        @output(id="ui_var1")
        @render.ui
        def ui_var1():
            isCat = get_data()["var_types"]["isCat"]
            return ui.input_select(
                id="var1",
                label="Select a categorical variable:",
                selected=self.state.get("var1", None),
                choices=isCat,
            )

        @output(id="ui_var2")
        @render.ui
        def ui_var2():
            isBin = get_data()["var_types"]["isBin"].copy()
            return ui.input_select(
                id="var2",
                label="Categorical Variable (binary):",
                selected=self.state.get("var2", None),
                choices=isBin,
            )

        @output(id="ui_lev")
        @render.ui
        def ui_lev():
            levs = list(get_data()["data"][input.var2()].unique())
            return ui.input_select(
                id="lev",
                label="Choose level:",
                selected=self.state.get("lev", levs[0]),
                choices=levs,
            )

        def combo_choices():
            data = get_data()["data"].copy()
            levels = data[input.var1()].astype("category").cat.categories
            return list(ru.iterms(levels))

        @output(id="ui_combos")
        @render.ui
        def ui_combos():
            choices = combo_choices()
            return ui.input_select(
                id="comb",
                label="Choose combinations:",
                selected=self.state.get("comb", None),
                choices=choices,
                multiple=True,
                size=min(3, len(choices)),
            )

        def estimation_code():
            data_name, code = (get_data()[k] for k in ["data_name", "code"])
            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "var1": input.var1(),
                "var2": input.var2(),
                "lev": input.lev(),
                "comb": list(input.comb()),
                "alt_hyp": input.alt_hyp(),
                "conf": input.conf(),
                "adjust": ifelse(input.adjust() == "None", None, input.adjust()),
            }

            args_string = ru.drop_default_args(args, rsm.basics.compare_props)
            return f"""rsm.basics.compare_props({args_string})""", code

        show_code, estimate = mu.make_estimate(
            self,
            input,
            output,
            get_data,
            fun="basics.compare_props",
            ret="cp",
            ec=estimation_code,
            run=False,
            debug=True,
        )

        def summary_code():
            args = {"extra": input.extra()}
            args_string = ru.drop_default_args(args, rsm.basics.compare_props.summary)
            return f"""cp.summary({args_string})"""

        mu.make_summary(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="cp",
            sum_fun=rsm.basics.compare_props.summary,
            sc=summary_code,
        )

        def plot_code():
            args = {"plots": input.plots()}
            args_string = ru.drop_default_args(args, rsm.basics.compare_props.plot)
            return f"""cp.plot({args_string})"""

        mu.make_plot(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="cp",
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


def compare_props(
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
    Launch a Radiant-for-Python app for compare props hypothesis testing
    """
    host = ru.set_host(host)
    if data_dct is None:
        data_dct, descriptions_dct = ru.get_dfs(pkg="data", name="titanic")
    rc = basics_compare_props(data_dct, descriptions_dct, state=state, code=code)
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
    compare_props(debug=False)
