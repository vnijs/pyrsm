from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import io, os, signal
import pyrsm as rsm
from contextlib import redirect_stdout
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
                width=3,
            ),
        ),
    )


def ui_plot():
    return (
        ui.panel_conditional(
            "input.tabs == 'Plot'",
            ui.panel_well(
                ui.input_select(
                    id="plots",
                    label="Plots",
                    selected=None,
                    choices={
                        "None": "None",
                        "hist": "Histogram",
                        "sim": "Simulate",
                    },
                ),
                width=3,
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


class basics_single_mean:
    def __init__(self, datasets: dict, descriptions=None) -> None:
        ru.init(self, datasets, descriptions=descriptions)

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
                        ui_plot(),
                    ),
                    ui.column(8, ui_main()),
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
        @reactive.Calc
        def get_data():
            return ru.get_data(input, self)

        ru.make_data_outputs(self, input, output)

        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)

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
            return f"""{code}\nsm = rsm.single_mean(data={data_name}, var="{input.var()}", alt_hyp="{input.alt_hyp()}", conf={input.conf()}, comp_value={input.comp_value()})"""

        @reactive.Calc
        def single_mean():
            data, data_name = (get_data()[k] for k in ["data", "data_name"])
            return rsm.single_mean(
                data={data_name: data},
                var=input.var(),
                alt_hyp=input.alt_hyp(),
                conf=input.conf(),
                comp_value=input.comp_value(),
            )

        def summary_code():
            return f"""sm.summary()"""

        @output(id="show_summary_code")
        @render.text
        def show_summary_code():
            cmd = f"""{estimation_code()}\n{summary_code()}"""
            return ru.code_formatter(cmd, self)

        @output(id="summary")
        @render.text
        def summary():
            out = io.StringIO()
            with redirect_stdout(out):
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
                cmd = f"""{estimation_code()}\n{plot_code()}"""
                return ru.code_formatter(cmd, self)

        @output(id="plot")
        @render.plot
        def plot():
            plots = input.plots()
            if plots != "None":
                sm = single_mean()  # get reactive object into local scope
                cmd = f"""{plot_code()}"""
                return eval(cmd)


demand_uk, demand_uk_description = rsm.load_data(pkg="basics", name="demand_uk")
rc = basics_single_mean({"demand_uk": demand_uk}, {"demand_uk": demand_uk_description})
app = App(rc.shiny_ui(), rc.shiny_server, debug=True)
