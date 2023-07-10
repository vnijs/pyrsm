from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import webbrowser, nest_asyncio, uvicorn
import io
import pyrsm as rsm
from contextlib import redirect_stdout, redirect_stderr
import pyrsm.radiant.utils as ru


def ui_summary():
    return (
        ui.panel_conditional(
            "input.tabs == 'Summary'",
            ui.panel_well(
                ui.output_ui("ui_var"),
                ui.input_select(
                    id="alt_hyp",
                    label="Alternative hypothesis:",
                    selected="two-sided",
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
                    value=0.95,
                ),
                ui.input_numeric(
                    id="comp_value",
                    label="Comparison value:",
                    value=0,
                ),
            ),
        ),
    )


choices = {
    "None": "None",
    "hist": "Histogram",
    "sim": "Simulate",
}


class basics_single_mean:
    def __init__(self, datasets: dict, descriptions=None, open=True) -> None:
        ru.init(self, datasets, descriptions=descriptions, open=open)

    def shiny_ui(self):
        return ui.page_navbar(
            ru.head_content(),
            ui.nav(
                "Basics > Single mean",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ui_summary(),
                        ru.ui_plot(choices),
                    ),
                    ui.column(8, ru.ui_main_basics()),
                ),
            ),
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/basics-single-mean.ipynb",
                "Single mean example notebook",
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
        @output(id="ui_var")
        @render.ui
        def ui_var():
            isNum = get_data()["var_types"]["isNum"]
            return ui.input_select(
                id="var",
                label="Variable (select one)",
                selected=None,
                choices=isNum,
            )

        def estimation_code():
            data_name, code = (get_data()[k] for k in ["data_name", "code"])

            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "var": input.var(),
                "alt_hyp": input.alt_hyp(),
                "conf": input.conf(),
                "comp_value": input.comp_value(),
            }

            args_string = ru.drop_default_args(args, rsm.basics.single_mean)
            return f"""rsm.basics.single_mean({args_string})""", code

        def show_code():
            sc = estimation_code()
            return f"""{sc[1]}\nsm = {sc[0]}"""

        @reactive.Calc
        def single_mean():
            locals()[input.datasets()] = self.datasets[
                input.datasets()
            ]  # get data into local scope
            return eval(estimation_code()[0])

        def summary_code():
            return f"""sm.summary()"""

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
                sm = single_mean()  # get the reactive object into local scope
                sm.summary()
            return out.getvalue()

        def plot_code():
            return f"""sm.plot("{input.plots()}")"""

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
                sm = single_mean()  # get reactive object into local scope
                cmd = f"""{plot_code()}"""
                return eval(cmd)


def single_mean(
    data_dct: dict,
    descriptions_dct: dict = None,
    open: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Radiant-for-Python app for single_mean hypothesis testing
    """
    rc = basics_single_mean(data_dct, descriptions_dct, open)
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

    demand_uk, demand_uk_description = rsm.load_data(pkg="basics", name="demand_uk")
    single_mean(
        {"demand_uk": demand_uk}, {"demand_uk": demand_uk_description}, open=True
    )
