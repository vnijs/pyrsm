from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import webbrowser, nest_asyncio, uvicorn
import signal, os, sys, tempfile
import pyrsm as rsm
from contextlib import redirect_stdout, redirect_stderr
import pyrsm.radiant.utils as ru
import pyrsm.radiant.model_utils as mu

choices = {
    "observed": "Observed",
    "expected": "Expected",
    "chisq": "Chi-squared",
    "dev_std": "Deviation std.",
    "perc_row": "Row percentages",
    "perc_col": "Column percentages",
    "perc": "Table percentages",
}


def ui_summary():
    return ui.panel_conditional(
        "input.tabs == 'Summary'",
        ui.panel_well(
            ui.output_ui("ui_var1"),
            ui.output_ui("ui_var2"),
            ui.input_checkbox_group(
                id="output",
                label="Select output tables:",
                choices=choices,
            ),
        ),
    )


plots = {"None": "None"}
plots.update(choices)


class basics_cross_tabs:
    def __init__(self, datasets: dict, descriptions=None, code=True) -> None:
        ru.init(self, datasets, descriptions=descriptions, code=code)

    def shiny_ui(self, *args):
        return ui.page_navbar(
            ru.head_content(),
            ui.nav(
                "<< Basics > Cross-tabs >>",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ui_summary(),
                        ru.ui_plot(plots),
                    ),
                    ui.column(8, ru.ui_main_basics()),
                ),
            ),
            *args,
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/basics-cross-tabs.ipynb",
                "Cross-tabs example notebook",
            ),
            ru.ui_stop(),
            title="Radiant for Python",
            inverse=True,
            id="navbar_id",
        )

    def shiny_server(self, input: Inputs, output: Outputs, session: Session):
        # --- section standard for all apps ---
        get_data = ru.make_data_elements(self, input, output, session)

        # --- section unique to each app ---
        @output(id="ui_var1")
        @render.ui
        def ui_var1():
            isCat = get_data()["var_types"]["isCat"]
            return ui.input_select(
                id="var1",
                label="Select a categorical variable:",
                selected=None,
                choices=isCat,
            )

        @output(id="ui_var2")
        @render.ui
        def ui_var2():
            isCat = get_data()["var_types"]["isCat"]
            if (input.var1() is not None) and (input.var1() in isCat):
                del isCat[input.var1()]

            return ui.input_select(
                id="var2",
                label="Select a categorical variable:",
                selected=None,
                choices=isCat,
            )

        def estimation_code():
            data_name, code = (get_data()[k] for k in ["data_name", "code"])

            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "var1": input.var1(),
                "var2": input.var2(),
            }

            args_string = ru.drop_default_args(args, rsm.basics.cross_tabs)
            return f"""rsm.basics.cross_tabs({args_string})""", code

        show_code, estimate = mu.make_estimate(
            self,
            input,
            output,
            get_data,
            fun="basics.cross_tabs",
            ret="ct",
            ec=estimation_code,
            run=False,
            debug=True,
        )

        def summary_code():
            args = [c for c in input.output()]
            return f"""ct.summary(output={args})"""

        mu.make_summary(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="ct",
            sum_fun=rsm.basics.cross_tabs.summary,
            sc=summary_code,
        )

        def plot_code():
            return f"""ct.plot(plots="{input.plots()}")"""

        mu.make_plot(
            self,
            input,
            output,
            session,
            show_code,
            estimate,
            ret="ct",
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


def cross_tabs(
    data_dct: dict = None,
    descriptions_dct: dict = None,
    code: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Radiant-for-Python app for cross-tabs hypothesis testing
    """
    if data_dct is None:
        data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
    rc = basics_cross_tabs(data_dct, descriptions_dct, code=code)
    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    print(f"Listening on http://{host}:{port}")
    ru.message()

    # redirect stdout and stderr to the temporary file
    temp = tempfile.NamedTemporaryFile()
    sys.stdout = open(temp.name, "w")
    sys.stderr = open(temp.name, "w")

    uvicorn.run(
        App(rc.shiny_ui(), rc.shiny_server),
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    cross_tabs()