from shiny import App, render, ui, reactive, Inputs, Outputs, Session
from faicons import icon_svg
import io, os, signal, black
from pathlib import Path
import pyrsm as rsm
import pandas as pd
from contextlib import redirect_stdout
from pyrsm.radiant.utils import *


class basics_cross_tabs:
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
                        "input.tabs_cross_tabs == 'Data'",
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
                        "input.tabs_cross_tabs == 'Summary'",
                        ui.panel_well(
                            ui.output_ui("ui_var_1"),
                            ui.output_ui("ui_var_2"),
                            ui.input_checkbox_group(
                                id="sel_func",
                                label="Select functions",
                                choices={
                                    "observed": "Observed",
                                    "expected": "Expected",
                                    "chisq": "Chi-squared",
                                    "dev_std": "Deviation std.",
                                    "perc_row": "Row percentages",
                                    "perc_col": "Column percentages",
                                    "perc": "Table percentages",
                                },
                            ),                            
                            width=3,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.tabs_cross_tabs == 'Plot'",
                        ui.panel_well(
                            ui.input_select(
                                id="plots",
                                label="Plots",
                                selected=None,
                                choices={
                                    "observed": "Observed",
                                    "expected": "Expected",
                                    "chisq": "Chi-squared",
                                    "dev_std": "Deviation std.",
                                    "perc_row": "Row percentages",
                                    "perc_col": "Column percentages",
                                    "perc": "Table percentages",
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
                            ui.output_text_verbatim("cross_tabs_summary"),
                        ),
                        ui.nav(
                            "Plot",
                            ui.output_ui("show_plot_code"),
                            ui.output_plot("cross_tabs_plot", height="600px"),
                        ),
                        id="tabs_cross_tabs",
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

        @output(id="ui_var_1")
        @render.ui
        def ui_var_1():
            isCat = get_data()[2]["isCat"]
            return ui.input_select(
                id="var_1",
                label="Select a categorical variable",
                selected=None,
                choices=isCat,
            )
        @output(id="ui_var_2")
        @render.ui
        def ui_var_2():
            isCat = get_data()[2]["isCat"]
            return ui.input_select(
                id="var_2",
                label="Select a categorical variable",
                selected=None,
                choices=isCat,
            )
        def estimation_code():
            var1 = input.var_1()
            var2 = input.var_2()
            fname, _, code = get_data()[1:]
            return f"""{code}\nct = rsm.cross_tabs(df={fname}, var1="{var1}", var2="{var2}")"""

        @reactive.Calc
        def cross_tabs():
            data = get_data()[0]
            return rsm.cross_tabs(
                df=data,
                var1 = input.var_1(),
                var2 = input.var_2(),
            )

        def summary_code():
            return f"""sm.summary()"""

        @output(id="show_summary_code")
        @render.text
        def show_summary_code():
            cmd = f"""{estimation_code()}\n{summary_code()}"""
            return code_formatter(cmd)

        @output(id="cross_tabs_summary")
        @render.text
        def cross_tabs_summary():
            out = io.StringIO()
            with redirect_stdout(out):
                sm = cross_tabs()  # get the reactive object into local scope
                sm.summary()
            return out.getvalue()

        def plot_code():
            return f"""ct.plot("{input.plots()}")"""

        @output(id="show_plot_code")
        @render.text
        def show_plot_code():
            plots = input.plots()
            if plots != "None":
                cmd = f"""{estimation_code()}\n{plot_code()}"""
                return code_formatter(cmd)

        @output(id="cross_tabs_plot")
        @render.plot
        def cross_tabs_plot():
            plots = input.plots()
            if plots != "None":
                ct = cross_tabs()  # get reactive object into local scope
                cmd = f"""{plot_code()}"""
                return eval(cmd)

        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)


rsm.load_data(pkg="basics", name="newspaper", dct=globals())
mr = basics_cross_tabs({"newspaper": newspaper})
app = App(mr.shiny_ui(), mr.shiny_server)