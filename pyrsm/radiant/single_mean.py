from shiny import App, render, ui, reactive, Inputs, Outputs, Session
from faicons import icon_svg
import webbrowser, nest_asyncio, uvicorn
import io, os, signal, black
from pathlib import Path
import pyrsm as rsm
import pandas as pd
from contextlib import redirect_stdout
from pyrsm.radiant.utils import *


class basics_single_mean:
    def __init__(self, datasets) -> None:
        self.datasets = rsm.ifelse(
            isinstance(datasets, dict), datasets, {"dataset": datasets}
        )
        self.dataset_list = list(datasets.keys())
        self.stop_code = ""

    def shiny_ui(self):
        return ui.page_fluid(
            ui.head_content(
                ui.tags.script(
                    (Path(__file__).parent / "www/returnTextAreaBinding.js").read_text()
                ),
                ui.tags.script((Path(__file__).parent / "www/copy.js").read_text()),
                ui.tags.style((Path(__file__).parent / "www/style.css").read_text()),
                # ui.tags.style("style.css"),  # not getting picked up for some reason
            ),
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.panel_well(
                        ui.input_select("datasets", "Datasets:", self.dataset_list),
                        width=3,
                    ),
                    ui.panel_conditional(
                        "input.tabs_regress == 'Data'",
                        ui.panel_well(
                            ui.input_checkbox("show_filter", "Show data filter"),
                            ui.panel_conditional(
                                "input.show_filter == true",
                                input_return_text_area(
                                    "data_filter",
                                    "Data Filter:",
                                    rows=2,
                                    placeholder="Provide a filter (e.g., price >  5000) and press return",
                                ),
                                input_return_text_area(
                                    "data_arrange",
                                    "Data arrange (sort):",
                                    rows=2,
                                    placeholder="Arrange (e.g., ['color', 'price'], ascending=[True, False])) and press return",
                                ),
                                input_return_text_area(
                                    "data_slice",
                                    "Data slice (rows):",
                                    rows=1,
                                    placeholder="e.g., 1:50 and press return",
                                ),
                            ),
                        ),
                    ),
                    ui.panel_conditional(
                        "input.tabs_single_mean == 'Summary'",
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
                    ui.panel_conditional(
                        "input.tabs_single_mean == 'Plot'",
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
                    ui.panel_well(
                        ui.input_action_button(
                            "stop",
                            "Stop app",
                            icon=icon_svg("stop"),
                            class_="btn-danger",
                            width="100%",
                            onclick="window.close();",
                        ),
                    ),
                ),
                ui.panel_main(
                    ui.navset_tab_card(
                        ui.nav(
                            "Data",
                            ui.output_ui("show_data_code"),
                            ui.output_ui("show_data"),
                            ui.output_ui("show_description"),
                        ),
                        ui.nav(
                            "Summary",
                            ui.output_ui("show_summary_code"),
                            ui.output_text_verbatim("single_mean_summary"),
                        ),
                        ui.nav(
                            "Plot",
                            ui.output_ui("show_plot_code"),
                            ui.output_plot("single_mean_plot", height="600px"),
                        ),
                        id="tabs_single_mean",
                    )
                ),
            ),
        )

    def shiny_server(self, input: Inputs, output: Outputs, session: Session):
        def code_formatter(code):
            cmd = self.stop_code = black.format_str(code, mode=black.Mode())
            return ui.HTML(
                f"""<pre><details><summary>Code</summary>{cmd}</details></pre>"""
            )
            # needs syntax highlighting but ui.markdown doesn't provide that yet
            # return ui.markdown(f"""```python\n{cmd}\n```""")
            # cmd = Markdown(f"\nGenerated code:\n\n```python\n{cmd}\n```")

        @reactive.Calc
        def get_data():
            data_name = input.datasets()
            data = self.datasets[data_name]
            if input.show_filter():
                if not is_empty(input.data_filter()):
                    data = data.query(input.data_filter())
                if not is_empty(input.data_arrange()):
                    data = eval(f"""data.sort_values({input.data_arrange()})""")
                if not is_empty(input.data_slice()):
                    data = eval(f"""data.iloc[{input.data_slice()}, :]""")

            types = {c: [data[c].dtype, data[c].nunique()] for c in data.columns}
            isNum = {
                c: f"{c} ({t[0].name})"
                for c, t in types.items()
                if pd.api.types.is_numeric_dtype(t[0])
            }
            isBin = {c: f"{c} ({t[0].name})" for c, t in types.items() if t[1] == 2}
            isCat = {
                c: f"{c} ({t[0].name})"
                for c, t in types.items()
                if c in isBin or pd.api.types.is_categorical_dtype(t[0]) or t[1] < 10
            }
            var_types = {
                "all": {c: f"{c} ({t[0].name})" for c, t in types.items()},
                "isNum": isNum,
                "isBin": isBin,
                "isCat": isCat,
            }

            return (
                data,
                data_name,
                var_types,
                f"import pyrsm as rsm\n# {data_name} = pd.read_pickle('{data_name}.pkl')",
            )

        @output(id="show_data_code")
        @render.ui
        def show_data_code():
            fname, _, code = get_data()[1:]
            if input.show_filter():
                code += f"""\n{fname} = {fname}"""
                if not is_empty(input.data_filter()):
                    code += f""".query("{input.data_filter()}")"""
                if not is_empty(input.data_arrange()):
                    code += f""".sort_values({input.data_arrange()})"""
                if not is_empty(input.data_slice()):
                    code += f""".iloc[{input.data_slice()}, :]"""
            return code_formatter(code)

        @output(id="show_data")
        @render.ui
        def show_data():
            data = get_data()[0]
            return (
                ui.HTML(
                    data.head(10).to_html(
                        classes="table table-striped data_preview", index=False
                    ),
                ),
                ui.p(f"Showing 10 rows out of {data.shape[0]:,}"),
            )

        @output(id="show_description")
        @render.ui
        def show_description():
            data = get_data()[0]
            if hasattr(data, "description"):
                return ui.markdown(get_data()[0].description)
            else:
                return ui.h3("No data description available")

        @output(id="ui_var")
        @render.ui
        def ui_var():
            isNum = get_data()[2]["isNum"]
            return ui.input_select(
                id="var",
                label="Variable (select one)",
                selected=None,
                choices=isNum,
            )

        def estimation_code():
            var = input.var()
            fname, _, code = get_data()[1:]
            return f"""{code}\nsm = rsm.single_mean(data={fname}, var="{var}", alt_hyp="{input.alt_hyp()}", conf={input.conf()}, comp_value={input.comp_value()})"""

        @reactive.Calc
        def single_mean():
            data = get_data()[0]
            return rsm.single_mean(
                data=data,
                var=input.var(),
                alt_hyp=input.alt_hyp(),
                conf=input.conf(),
                comp_value=input.comp_value(),
            )

        def summary_code(shiny=False):
            return f"""sm.summary()"""

        @output(id="show_summary_code")
        @render.text
        def show_summary_code():
            cmd = f"""{estimation_code()}\n{summary_code()}"""
            return code_formatter(cmd)

        @output(id="single_mean_summary")
        @render.text
        def single_mean_summary():
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
                return code_formatter(cmd)

        @output(id="single_mean_plot")
        @render.plot
        def single_mean_plot():
            plots = input.plots()
            if plots != "None":
                sm = single_mean()  # get reactive object into local scope
                cmd = f"""{plot_code()}"""
                return eval(cmd)

        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)


def single_mean(
    data_dct: dict,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Shiny-for-Python app for single_mean hypothesis testing
    """
    sm = basics_single_mean(data_dct)
    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    print(f"Listening on http://{host}:{port}")
    print(
        "Pyrsm and Radiant are opensource tools and free to use. If you\nare a student or instructor using pyrsm or Radiant for a class,\nas a favor to the developers, please send an email to\n<radiant@rady.ucsd.edu> with the name of the school and class."
    )
    uvicorn.run(
        App(sm.shiny_ui(), sm.shiny_server),
        host=host,
        port=port,
        log_level=log_level,
    )
