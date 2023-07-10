from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import webbrowser, nest_asyncio, uvicorn
import io, os, signal
import pyrsm as rsm
from contextlib import redirect_stdout, redirect_stderr
import pyrsm.radiant.utils as ru

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
                id="select_output",
                label="Select output tables:",
                choices=choices,
            ),
        ),
    )


plots = {"None": "None"}
plots.update(choices)


class basics_cross_tabs:
    def __init__(self, datasets: dict, descriptions=None, open=True) -> None:
        ru.init(self, datasets, descriptions=descriptions, open=open)

    def shiny_ui(self):
        return ui.page_navbar(
            ru.head_content(),
            ui.nav(
                "Basics > Cross-tabs",
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
        get_data, stop_app = ru.standard_reactives(self, input, session)
        ru.make_data_outputs(self, input, output)

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

        def show_code():
            sc = estimation_code()
            return f"""{sc[1]}\nct = {sc[0]}"""

        @reactive.Calc
        def cross_tabs():
            locals()[input.datasets()] = self.datasets[
                input.datasets()
            ]  # get data into local scope
            return eval(estimation_code()[0])

        def summary_code():
            args = [c for c in input.select_output()]
            return f"""ct.summary(output={args})"""

        @output(id="show_summary_code")
        @render.text
        def show_summary_code():
            cmd = f"""{show_code()}\n{summary_code()}"""
            return ru.code_formatter(cmd, self)

        @output(id="summary")
        @render.text
        def summary():
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                ct = cross_tabs()  # get the reactive object into local scope
                eval(summary_code())
            return out.getvalue()

        def plot_code():
            return f"""ct.plot(output="{input.plots()}")"""

        @output(id="show_plot_code")
        @render.text
        def show_plot_code():
            plots = input.plots()
            if plots != "None":
                cmd = f"""{show_code()}\n{plot_code()}"""
                return ru.code_formatter(cmd, self)

        @output(id="plot")
        @render.plot
        def plot():
            plots = input.plots()
            if plots != "None":
                ct = cross_tabs()  # get reactive object into local scope
                cmd = f"""{plot_code()}"""
                return eval(cmd)


def cross_tabs(
    data_dct: dict,
    descriptions_dct: dict = None,
    open: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Radiant-for-Python app for linear regression analysis
    """
    rc = basics_cross_tabs(data_dct, descriptions_dct, open=open)
    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    print(f"Listening on http://{host}:{port}")
    print(
        "Pyrsm and Radiant are open source tools and free to use. If you\nare a student or instructor using pyrsm or Radiant for a class,\nas a favor to the developers, please send an email to\n<radiant@rady.ucsd.edu> with the name of the school and class."
    )
    uvicorn.run(
        App(rc.shiny_ui(), rc.shiny_server),
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    import pyrsm as rsm

    newspaper, newspaper_description = rsm.load_data(name="newspaper")
    cross_tabs(
        {"newspaper": newspaper}, {"newspaper": newspaper_description}, open=True
    )
