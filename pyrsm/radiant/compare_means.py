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
import pyrsm.radiant.model_utils as mu


def ui_summary(self):
    return ui.panel_conditional(
        "input.tabs == 'Summary'",
        ui.panel_well(
            ui.input_radio_buttons(
                id="var_type",
                label="Variable type:",
                selected=self.state.get("var_type", "categorical"),
                choices=["categorical", "numeric"],
                inline=True,
            ),
            ui.panel_conditional(
                "input.var_type == 'categorical'",
                ui.output_ui("ui_cvar1"),
                ui.output_ui("ui_cvar2"),
            ),
            ui.panel_conditional(
                "input.var_type == 'numeric'",
                ui.output_ui("ui_nvar1"),
                ui.output_ui("ui_nvar2"),
            ),
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
                id="sample_type",
                label="Sample Type:",
                selected=self.state.get("sample_type", "independent"),
                choices={
                    "independent": "independent",
                    "paired": "paired",
                },
                inline=True,
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
            ui.input_radio_buttons(
                id="test_type",
                label="Test type:",
                selected=self.state.get("test_type", "t-test"),
                choices={
                    "t-test": "t-test",
                    "wilcox": "Wilcox",
                },
                inline=True,
            ),
        ),
    )


choices = {
    "None": "None",
    "scatter": "Scatter plot",
    "density": "Density plot",
    "box": "Box plot",
    "bar": "Bar chart",
}


def plots_extra(self):
    return (
        ui.panel_conditional(
            "input.plots == 'scatter'",
            ui.input_select(
                "nobs",
                "Number of data points plotted:",
                selected=self.state.get("nobs", "1,000"),
                choices={1_000: "1,000", -1: "All"},
            ),
        ),
    )


class basics_compare_means:
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
                "Basics > Compare means",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ui_summary(self),
                        ru.ui_plot(self, choices, plots_extra(self)),
                    ),
                    ui.column(8, ru.ui_main_basics(self)),
                ),
            ),
            self.navbar,
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/basics-compare-means.ipynb",
                "Compare means example notebook",
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

        # --- section unique to each app ---
        @output(id="ui_cvar1")
        @render.ui
        def ui_cvar1():
            isCat = get_data()["var_types"]["isCat"]
            return ui.input_select(
                id="cvar1",
                label="Select a categorical variable:",
                selected=self.state.get("cvar1", None),
                choices=isCat,
            )

        @output(id="ui_cvar2")
        @render.ui
        def ui_cvar2():
            isNum = get_data()["var_types"]["isNum"].copy()
            return ui.input_select(
                id="cvar2",
                label="Numeric Variable:",
                selected=self.state.get("cvar2", None),
                choices=isNum,
            )

        @output(id="ui_nvar1")
        @render.ui
        def ui_nvar1():
            isNum = get_data()["var_types"]["isNum"]
            return ui.input_select(
                id="nvar1",
                label="Select a numeric variable:",
                selected=self.state.get("nvar1", None),
                choices=isNum,
            )

        @output(id="ui_nvar2")
        @render.ui
        def ui_nvar2():
            isNum = get_data()["var_types"]["isNum"].copy()
            if input.nvar1() is not None:
                del isNum[input.nvar1()]
            return ui.input_select(
                id="nvar2",
                label="Numeric Variable:",
                selected=self.state.get("nvar2", None),
                choices=isNum,
                multiple=True,
            )

        def combo_choices():
            data = get_data()["data"].copy()
            if input.var_type() == "categorical":
                levels = data[input.cvar1()].astype("category").cat.categories
            else:
                data = data.loc[:, [input.nvar1()] + list(input.nvar2())].melt()
                levels = data["variable"].unique()

            return list(ru.iterms(levels))

        @output(id="ui_combos")
        @render.ui
        def ui_combos():
            req(input.var_type())
            return ui.input_select(
                id="comb",
                label="Choose combinations:",
                selected=self.state.get("comb", None),
                choices=combo_choices(),
                multiple=True,
            )

        def estimation_code():
            data_name, code = (get_data()[k] for k in ["data_name", "code"])
            if input.var_type() == "categorical":
                var1 = input.cvar1()
                var2 = input.cvar2()
            else:
                var1 = input.nvar1()
                var2 = input.nvar2()

            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "var1": var1,
                "var2": var2,
                "comb": list(input.comb()),
                "alt_hyp": input.alt_hyp(),
                "conf": input.conf(),
                "sample_type": input.sample_type(),
                "adjust": ifelse(input.adjust() == "None", None, input.adjust()),
                "test_type": input.test_type(),
            }

            args_string = ru.drop_default_args(args, rsm.basics.compare_means)
            return f"""rsm.basics.compare_means({args_string})""", code

        show_code, estimate = mu.make_estimate(
            self,
            input,
            output,
            get_data,
            fun="basics.compare_means",
            ret="cm",
            ec=estimation_code,
            run=False,
            debug=True,
        )

        def summary_code():
            args = {"extra": input.extra()}
            args_string = ru.drop_default_args(args, rsm.basics.compare_means.summary)
            return f"""cm.summary({args_string})"""

        mu.make_summary(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="cm",
            sum_fun=rsm.basics.compare_means.summary,
            sc=summary_code,
        )

        def plot_code():
            args = {"plots": input.plots(), "nobs": input.nobs()}
            args_string = ru.drop_default_args(
                args, rsm.basics.compare_means.plot, ignore=["nobs"]
            )
            return f"""cm.plot({args_string})"""

        mu.make_plot(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="cm",
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


def compare_means(
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
    Launch a Radiant-for-Python app for compare means hypothesis testing
    """
    if data_dct is None:
        data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="salary")
    rc = basics_compare_means(data_dct, descriptions_dct, state=state, code=code)
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
    compare_means(debug=False)
