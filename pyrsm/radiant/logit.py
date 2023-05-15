from pathlib import Path
import numpy as np
import pandas as pd
import pyrsm as rsm
import io
from pathlib import Path
from contextlib import redirect_stdout
from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import black
import pyperclip

## next steps

## add a close button (navbar?) and return something (code?) on close
# https://shiny.rstudio.com/py/api/Session.html#shiny.Session
## adjust plot window height
## render table https://shiny.rstudio.com/py/api/ui.output_table.html#shiny.ui.output_table
## shown description as markdown https://shinylive.io/py/examples/#extra-packages


class model_logit:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset

    def shiny_ui(self):
        return ui.page_fluid(
            ui.head_content(
                ui.tags.script((Path(__file__).parent / "www/copy.js").read_text()),
                ui.tags.style((Path(__file__).parent / "www/style.css").read_text()),
                # ui.tags.script("copy.js"), # not getting picked up for some reason
                # ui.tags.style("style.css"), # not getting picked up for some reason
                # ui.tags.script(
                #     src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
                # ),
                # ui.tags.script(
                #     "if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub]);"
                # ),
            ),
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.panel_well(
                        ui.input_file(
                            id="logistic_data",
                            label="Upload a Pickle file",
                            accept=[".pkl"],
                            multiple=False,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.tabs_logistic == 'Summary'",
                        ui.panel_well(
                            ui.input_action_button(
                                "run",
                                "Estimate model",
                                class_="btn-success",
                                width="100%",
                            ),
                        ),
                        ui.panel_well(
                            ui.output_ui("ui_rvar"),
                            ui.output_ui("ui_evar"),
                            width=3,
                        ),
                    ),
                    ui.panel_conditional(
                        "input.tabs_logistic == 'Plot'",
                        ui.panel_well(
                            ui.input_action_button(
                                "plot",
                                "Create plot",
                                class_="btn-success",
                                width="100%",
                            ),
                        ),
                        ui.panel_well(
                            ui.input_select(
                                id="logistic_plots",
                                label="Plots",
                                selected=None,
                                choices={
                                    "or": "OR plot",
                                    "pred": "Prediction plot",
                                    "vimp": "Permutation importance",
                                },
                            ),
                            width=3,
                        ),
                    ),
                ),
                ui.panel_main(
                    ui.navset_tab_card(
                        ui.nav(
                            "Data",
                            ui.output_ui("show_data"),
                            ui.output_ui("show_data_code"),
                        ),
                        ui.nav(
                            "Summary",
                            ui.output_text_verbatim("logistic_summary"),
                            ui.output_ui("show_summary_code"),
                        ),
                        ui.nav(
                            "Plot",
                            ui.output_plot("logistic_plot"),
                            ui.output_ui("show_plot_code"),
                        ),
                        id="tabs_logistic",
                    )
                ),
            ),
        )

    def shiny_server(self, input: Inputs, output: Outputs, session: Session):
        @reactive.Calc
        def load_data():
            if input.logistic_data() is None:
                # if not "mydata" in globals():
                #     return "Please upload a Pickle file", "", ""
                # else:
                #     data = globals()["mydata"]
                #     return data, "data", "data = pd.read_pickle('data.pkl')"
                return self.dataset, "data", "data = pd.read_pickle('data.pkl')"
            else:
                f = input.logistic_data()
                file_name = f[0]["name"]
                fname = file_name.split(".")[0]
                code = f"""{fname} = pd.read_pickle("{file_name}")"""
                return pd.read_pickle(f[0]["datapath"]), fname, code

        @output(id="show_data")
        @render.ui
        def show_data():
            if input.logistic_data() is None:
                return "Please upload a Pickle file"
            else:
                data, _, _ = load_data()
                return ui.HTML(
                    data.head().to_html(
                        classes="table table-striped data_preview", index=False
                    )
                )

        @output(id="show_data_code")
        @render.ui
        def show_data_code():
            if input.logistic_data() is None:
                return ""
            else:
                _, _, code = load_data()
                return code_formatter(code)

        @output(id="ui_rvar")
        @render.ui
        def ui_rvar():
            data, _, _ = load_data()
            if isinstance(data, str):
                df_cols = []
            else:
                df_cols = list(data.columns)
            return ui.input_select(
                id="rvar",
                label="Response Variable",
                selected=None,
                choices=df_cols,
            )

        @output(id="ui_evar")
        @render.ui
        def ui_evar():
            data, _, _ = load_data()
            if isinstance(data, str):
                df_cols = []
            else:
                df_cols = list(data.columns)
                if (input.rvar() is not None) and (input.rvar() in df_cols):
                    df_cols.remove(input.rvar())
            return ui.input_select(
                id="evar",
                label="Explanatory Variables",
                selected=None,
                choices=df_cols,
                multiple=True,
                selectize=False,
            )

        @reactive.Calc
        def logistic_regression():
            data, _, _ = load_data()
            return rsm.logistic(
                dataset=data, rvar=input.rvar(), evar=list(input.evar())
            )

        @output(id="logistic_summary")
        @render.text
        @reactive.event(input.run, ignore_none=True)
        def logistic_summary():
            out = io.StringIO()
            with redirect_stdout(out):
                logistic_regression().summary()
            return out.getvalue()

        def code_formatter(code):
            cmd = black.format_str(code, mode=black.Mode())
            pyperclip.copy(cmd)
            cmd = f"""<pre><details><summary>Code</summary>{cmd}</details></pre>"""
            return ui.HTML(cmd)

        @output(id="show_summary_code")
        @render.text
        @reactive.event(input.run, ignore_none=True)
        def show_summary_code():
            rvar = input.rvar()
            evar = list(input.evar())
            _, fname, code = load_data()
            cmd = f"""{code}\nlr = rsm.logistic(dataset={fname}, rvar="{rvar}", evar={evar})\nlr.summary()"""
            return code_formatter(cmd)

        @output(id="logistic_plot")
        @render.plot()
        @reactive.event(input.plot, ignore_none=True)
        def logistic_plot():
            return logistic_regression().plot(plots=input.logistic_plots())

        @output(id="show_plot_code")
        @render.text
        @reactive.event(input.plot, ignore_none=True)
        def show_plot_code():
            rvar = input.rvar()
            evar = list(input.evar())
            plot_type = input.logistic_plots()
            _, fname, code = load_data()
            cmd = f"""{code}\nlr = rsm.logistic(dataset={fname}, rvar="{rvar}", evar={evar})\nlr.plot("{plot_type}")"""
            return code_formatter(cmd)


## should work based on https://shinylive.io/py/examples/#static-content
## but doesn't
# www_dir = Path(__file__).parent / "www"
# app = App(app_ui, server, static_assets=www_dir)


## uncomment for development and testing
# rsm.load_data(pkg="data", name="titanic", dct=globals())
# sl = shiny_logit(titanic)
# app = App(sl.shiny_ui(), sl.shiny_server)

## uncomment for development and testing
# rsm.load_data(pkg="data", name="titanic", dct=globals())


# def launch_logit(data):
#     lr = model_logit(data)
#     nest_asyncio.apply()
#     webbrowser.open("http://127.0.0.1:8000")
#     uvicorn.run(App(lr.shiny_ui(), lr.shiny_server), port=8000)


# launch_logit(titanic)
