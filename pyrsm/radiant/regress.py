from shiny import App, render, ui, reactive, Inputs, Outputs, Session
from faicons import icon_svg
import webbrowser, nest_asyncio, uvicorn
import io, os, signal, black
from pathlib import Path
import pyrsm as rsm
import pandas as pd
from contextlib import redirect_stdout
from datetime import datetime
from itertools import combinations
from .utils import *

# from itables import to_html_datatable as DT


## next steps
## try qgrid for interactive data table


class model_regress:
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
                        "input.tabs_regress == 'Summary'",
                        ui.panel_well(
                            ui.input_action_button(
                                "run",
                                "Estimate model",
                                icon=icon_svg("play"),
                                class_="btn-success",
                                width="100%",
                            ),
                        ),
                        ui.panel_well(
                            ui.output_ui("ui_rvar"),
                            ui.output_ui("ui_evar"),
                            ui.input_radio_buttons(
                                "show_interactions",
                                "Interactions:",
                                choices={0: "None", 2: "2-way", 3: "3-way"},
                                inline=True,
                            ),
                            ui.panel_conditional(
                                "input.show_interactions > 0",
                                ui.output_ui("ui_interactions"),
                            ),
                            ui.output_ui("ui_evar_test"),
                            ui.input_checkbox_group(
                                "controls",
                                "Additional output:",
                                {
                                    "ci": "Confidence intervals",
                                    "ssq": "Sum of Squares",
                                    "vif": "VIF",
                                },
                            ),
                            width=3,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.tabs_regress == 'Predict'",
                        ui.panel_well(
                            ui.input_select(
                                "pred_datasets", "Prediction data:", self.dataset_list
                            ),
                            width=3,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.tabs_regress == 'Plot'",
                        ui.panel_well(
                            ui.input_select(
                                id="regress_plots",
                                label="Plots",
                                selected=None,
                                choices={
                                    "None": "None",
                                    "dist": "Distribution",
                                    "corr": "Correlation",
                                    "scatter": "Scatter",
                                    "dashboard": "Dashboard",
                                    "residual": "Residual vs Explanatory",
                                    "pred": "Prediction plots",
                                    "vimp": "Permutation importance",
                                    "coef": "Coefficient plot",
                                },
                            ),
                            ui.panel_conditional(
                                "input.regress_plots == 'corr'",
                                ui.input_select(
                                    "nobs",
                                    "Number of data points plotted:",
                                    {1000: "1,000", -1: "All"},
                                ),
                            ),
                            ui.panel_conditional(
                                "input.regress_plots == 'pred'",
                                ui.output_ui("ui_incl_evar"),
                                ui.output_ui("ui_incl_interactions"),
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
                            ui.output_text_verbatim("regress_summary"),
                        ),
                        ui.nav(
                            "Predict",
                            ui.output_ui("show_predict_code"),
                            ui.output_ui("regress_predict"),
                        ),
                        ui.nav(
                            "Plot",
                            ui.output_ui("show_plot_code"),
                            ui.output_plot("regress_plot", height="800px"),
                        ),
                        id="tabs_regress",
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

        @output(id="ui_rvar")
        @render.ui
        def ui_rvar():
            isNum = get_data()[2]["isNum"]
            return ui.input_select(
                id="rvar",
                label="Response Variable",
                selected=None,
                choices=isNum,
            )

        @output(id="ui_evar")
        @render.ui
        def ui_evar():
            vars = get_data()[2]["all"]
            if (input.rvar() is not None) and (input.rvar() in vars):
                del vars[input.rvar()]

            return ui.input_select(
                id="evar",
                label="Explanatory Variables",
                selected=None,
                choices=vars,
                multiple=True,
                size=min(8, len(vars)),
                selectize=False,
            )

        @output(id="ui_interactions")
        @render.ui
        def ui_interactions():
            if len(input.evar()) > 1:
                choices = []
                nway = int(input.show_interactions())
                isNum = rsm.intersect(
                    list(input.evar()), list(get_data()[2]["isNum"].keys())
                )
                if len(isNum) > 0:
                    choices += qterms(isNum, nway=int(input.show_interactions()[0]))
                choices += iterms(input.evar(), nway=nway)
                return ui.input_select(
                    id="interactions",
                    label="Interactions",
                    selected=None,
                    choices=choices,
                    multiple=True,
                    size=min(8, len(choices)),
                    selectize=False,
                )

        @output(id="ui_evar_test")
        @render.ui
        def ui_evar_test():
            choices = {e: e for e in input.evar()}
            if int(input.show_interactions()[0]) > 1:
                choices.update({e: e for e in input.interactions()})
            return ui.input_selectize(
                id="evar_test",
                label="Variables to test:",
                selected=None,
                choices=choices,
                multiple=True,
            )

        @output(id="ui_incl_evar")
        @render.ui
        def ui_incl_evar():
            if len(input.evar()) > 1:
                return ui.input_select(
                    id="incl_evar",
                    label="Explanatory variables to include:",
                    selected=None,
                    choices=input.evar(),
                    multiple=True,
                    size=min(8, len(input.evar())),
                    selectize=False,
                )

        @output(id="ui_incl_interactions")
        @render.ui
        def ui_incl_interactions():
            if len(input.evar()) > 1:
                nway = int(input.show_interactions())
                choices = iterms(input.evar(), nway=nway)
                return ui.input_select(
                    id="incl_interactions",
                    label="Interactions to include:",
                    selected=None,
                    choices=choices,
                    multiple=True,
                    size=min(8, len(choices)),
                    selectize=False,
                )

        def model_code():
            rvar = input.rvar()
            evar = list(input.evar())
            fname, _, code = get_data()[1:]
            if int(input.show_interactions()) > 0 and len(input.interactions()) > 0:
                return f"""{code}\nreg = rsm.regress(dataset={fname}, rvar="{rvar}", evar={evar}, int={list(input.interactions())})"""
            else:
                return f"""{code}\nreg = rsm.regress(dataset={fname}, rvar="{rvar}", evar={evar})"""

        @reactive.Calc
        @reactive.event(input.run, ignore_none=True)
        def regress():
            now = datetime.now().time().strftime("%H:%M:%S")
            print(f"Model estimated at: {now}")
            data = get_data()[0]
            if int(input.show_interactions()) > 0:
                return rsm.regress(
                    dataset=data,
                    rvar=input.rvar(),
                    evar=list(input.evar()),
                    int=list(input.interactions()),
                )
            else:
                return rsm.regress(
                    dataset=data,
                    rvar=input.rvar(),
                    evar=list(input.evar()),
                )

        def summary_code(shiny=False):
            ctrl = input.controls()
            cmd = f"""ci={"ci" in ctrl}, ssq={"ssq" in ctrl}, vif={"vif" in ctrl}"""
            if shiny:
                cmd += """, shiny=True"""
            return f"""reg.summary({cmd})"""

        @output(id="show_summary_code")
        @render.text
        def show_summary_code():
            cmd = f"""{model_code()}\n{summary_code()}"""
            return code_formatter(cmd)

        @output(id="regress_summary")
        @render.text
        def regress_summary():
            out = io.StringIO()
            with redirect_stdout(out):
                reg = regress()  # get model object into local scope
                cmd = f"""{summary_code(shiny=True)}"""
                eval(cmd)
            return out.getvalue()

        @output(id="show_predict_code")
        @render.text
        def show_predict_code():
            cmd = f"""{model_code()}\npred = reg.predict(df={input.pred_datasets()}, ci=True)"""
            return code_formatter(cmd)

        @output(id="regress_predict")
        @render.text
        def regress_predict():
            return ui.HTML(
                regress()
                .predict(df=self.datasets[input.pred_datasets()], ci=True)
                .round(3)
                .head(10)
                .to_html(classes="table table-striped data_preview", index=False)
            )

        def plot_code():
            plots = input.regress_plots()
            if plots == "pred":
                incl = list(input.incl_evar())
                incl_int = list(input.incl_interactions())
                cmd = f""", incl={incl}"""
                if len(incl_int) > 0:
                    cmd += f""", incl_int={incl_int}"""
                cmd = f"""reg.plot(plots="{plots}" {cmd})"""
            elif plots == "corr":
                cmd = f"""reg.plot(plots="{plots}", nobs={input.nobs()})"""
            else:
                cmd = f"""reg.plot(plots="{plots}")"""
            return cmd

        @output(id="show_plot_code")
        @render.text
        def show_plot_code():
            plots = input.regress_plots()
            if plots != "None":
                cmd = f"""{model_code()}\n{plot_code()}"""
                return code_formatter(cmd)

        def plot_height():
            plots = input.regress_plots()
            if plots == "pred":
                return "800px"
            else:
                return "1000px"

        @output(id="regress_plot")
        # @render.plot(height=plot_height)
        @render.plot
        def regress_plot():
            plots = input.regress_plots()
            if plots != "None":
                reg = regress()  # get model object into local scope
                cmd = f"""{plot_code()}"""
                return eval(cmd)

        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)


## should work based on https://shinylive.io/py/examples/#static-content
## but doesn't
# www_dir = Path(__file__).parent / "www"
# app = App(app_ui, server, static_assets=www_dir)


## uncomment for development and testing
# if __file__ == "app.py":
# if Path(__file__).name == "app.py":
# rsm.load_data(pkg="data", name="diamonds", dct=globals())
# mr = model_regress({"diamonds": diamonds, "diamonds100": diamonds.sample(100)})
# app = App(mr.shiny_ui(), mr.shiny_server)


def regress(
    data_dct: dict,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Shiny-for-Python app for regression analysis
    """
    mr = model_regress(data_dct)
    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    print(f"Listening on http://{host}:{port}")
    print(
        "Pyrsm and Radiant are opensource tools and free to use. If you\nare a student or instructor using pyrsm or Radiant for a class,\nas a favor to the developers, please send an email to\n<radiant@rady.ucsd.edu> with the name of the school and class."
    )
    uvicorn.run(
        App(mr.shiny_ui(), mr.shiny_server),
        host=host,
        port=port,
        log_level=log_level,
    )
