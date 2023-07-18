from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import webbrowser, nest_asyncio, uvicorn
import io, os, signal
import pyrsm as rsm
from contextlib import redirect_stdout
import pyrsm.radiant.utils as ru

choices = {
    "observed": "Observed",
    "expected": "Expected",
    "chisq": "Chi-squared",
    "dev_std": "Deviation std.",
}


def ui_summary():
    return ui.panel_conditional(
        "input.tabs == 'Summary'",
        ui.panel_well(
            ui.output_ui("ui_var"),
            ui.input_numeric(
                "prob_1", "Probability 1", value=0.9, min=0, max=1, step=0.01
            ),
            ui.input_numeric(
                "prob_2", "Probability 2", value=0.9, min=0, max=1, step=0.01
            ),
            ui.input_checkbox_group(
                id="select_output",
                label="Select output tables:",
                choices=choices,
            ),
        ),
    )


def ui_plot():
    plots = {"None": "None"}
    plots.update(choices)
    return (
        ui.panel_conditional(
            "input.tabs == 'Plot'",
            ui.panel_well(
                ui.input_select(
                    id="select_plot",
                    label="Select plot:",
                    choices=plots,
                ),
            ),
        ),
    )


def ui_main():
    return ui.navset_tab_card(
        ru.ui_data_main(),
        ui.nav(
            "Summary",
            ui.output_ui("show_summary_code"),
            ui.output_text_verbatim("summary"),
        ),
        ui.nav(
            "Plot",
            ui.output_ui("show_plot_code"),
            ui.output_plot("plot", height="500px", width="700px"),
        ),
        id="tabs",
    )


class basics_goodness:
    def __init__(self, datasets: dict, descriptions=None, open=True) -> None:
        ru.init(self, datasets, descriptions=descriptions, open=open)

    def shiny_ui(self):
        return ui.page_navbar(
            ru.head_content(),
            ui.nav(
                "Basics > Goodness-of-fit",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ui_summary(),
                        ui_plot(),
                    ),
                    ui.column(8, ru.ui_main_basics()),
                ),
            ),
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/basics-cross-tabs.ipynb",  # no example notebook for goodness of fit
                "Goodness-of-fit example notebook",
            ),
            ru.ui_stop(),
            title="Radiant for Python",
            inverse=True,
            id="navbar_id",
        )

    def shiny_server(self, input: Inputs, output: Outputs, session: Session):
        # --- section standard for all apps ---
        get_data = ru.make_data_elements(self, input, output)

        # --- section unique to each app ---
        @output(id="ui_var")
        @render.ui
        def ui_var1():
            isCat = get_data()["var_types"]["isCat"]
            return ui.input_select(
                id="var",
                label="Select a categorical variable:",
                selected=None,
                choices=isCat,
            )

        def estimation_code():
            data_name, code = (get_data()[k] for k in ["data_name", "code"])

            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "variable": input.var(),
                "probabilities": (input.prob_1(), input.prob_2()),
            }

            args_string = ru.drop_default_args(args, rsm.basics.goodness)
            return f"""rsm.basics.goodness({args_string})""", code

        def show_code():
            sc = estimation_code()
            return f"""{sc[1]}\nct = {sc[0]}"""

        @reactive.Calc
        def estimate():
            locals()[input.datasets()] = self.datasets[
                input.datasets()
            ]  # get data into local scope
            return eval(estimation_code()[0])

        def summary_code():
            args = [c for c in input.select_output()]
            return f"""gf.summary(output={args})"""

        @output(id="show_summary_code")
        @render.text
        def show_summary_code():
            cmd = f"""{show_code()}\n{summary_code()}"""
            return ru.code_formatter(cmd, self)

        @output(id="summary")
        @render.text
        def summary():
            out = io.StringIO()
            with redirect_stdout(out):
                gf = estimate()  # get the reactive object into local scope
                eval(summary_code())
            return out.getvalue()

        def plot_code():
            return f"""gf.plot(output="{input.select_plot()}")"""

        @output(id="show_plot_code")
        @render.text
        def show_plot_code():
            plots = input.select_plot()
            if plots != "None":
                cmd = f"""{show_code()}\n{plot_code()}"""
                return ru.code_formatter(cmd, self)

        @output(id="plot")
        @render.plot
        def plot():
            plots = input.select_plot()
            if plots != "None":
                gf = estimate()  # get reactive object into local scope
                cmd = f"""{plot_code()}"""
                return eval(cmd)

        # --- section standard for all apps ---
        # stops returning code if moved to utils
        @reactive.Effect
        @reactive.event(input.stop, ignore_none=true)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.sigterm)


def goodness(
    data_dct: dict,
    descriptions_dct: dict = None,
    open: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Radiant-for-Python app for goodness of fit analysis
    """
    rc = basics_goodness(data_dct, descriptions_dct, open=open)
    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    print(f"Listening on http://{host}:{port}")
    ru.message()
    uvicorn.run(
        App(rc.shiny_ui(), rc.shiny_server),
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    import pyrsm as rsm

    newspaper, newspaper_description = rsm.load_data(pkg="basics", name="newspaper")
    rc = basics_goodness(
        {"newspaper": newspaper}, {"newspaper": newspaper_description}, open=True
    )
    app = App(rc.shiny_ui(), rc.shiny_server)
