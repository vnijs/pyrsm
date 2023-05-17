import webbrowser, nest_asyncio, uvicorn
import io, os, signal, black
from pathlib import Path
import pyrsm as rsm
from contextlib import redirect_stdout
from shiny import App, render, ui, reactive, Inputs, Outputs, Session
from datetime import datetime

## next steps
## try qgrid for interactive data table
## shown description as markdown https://shinylive.io/py/examples/#extra-packages


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
                        "input.tabs_regress == 'Summary'",
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
                        ui.panel_well(
                            ui.input_checkbox_group(
                                "controls",
                                "Controls:",
                                {"ssq": "Sum of Squares", "vif": "VIF"},
                            )
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
                            width=3,
                        ),
                    ),
                    ui.panel_well(
                        ui.input_action_button(
                            "stop",
                            "Stop app",
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
        def load_data():
            return (
                self.datasets[input.datasets()],
                input.datasets(),
                f"{input.datasets()} = pd.read_pickle('{input.datasets()}.pkl')",
            )

        @output(id="show_data")
        @render.ui
        def show_data():
            data, _, _ = load_data()
            return ui.HTML(
                data.head().to_html(
                    classes="table table-striped data_preview", index=False
                )
            )

        @output(id="show_data_code")
        @render.ui
        def show_data_code():
            _, _, code = load_data()
            return code_formatter(code)

        @output(id="ui_rvar")
        @render.ui
        def ui_rvar():
            data, _, _ = load_data()
            df_cols = {c: f"{c} ({data[c].dtype})" for c in data.columns}
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
            df_cols = list(data.columns)
            if (input.rvar() is not None) and (input.rvar() in df_cols):
                df_cols.remove(input.rvar())

            df_cols = {c: f"{c} ({data[c].dtype})" for c in df_cols}
            return ui.input_select(
                id="evar",
                label="Explanatory Variables",
                selected=None,
                choices=df_cols,
                multiple=True,
                selectize=False,
            )

        @reactive.Calc
        @reactive.event(input.run, ignore_none=True)
        def regress():
            now = datetime.now().time().strftime("%H:%M:%S")
            print(f"Model estimated at: {now}")
            data, _, _ = load_data()
            return rsm.regress(dataset=data, rvar=input.rvar(), evar=list(input.evar()))

        @output(id="regress_summary")
        @render.text
        def regress_summary():
            out = io.StringIO()
            with redirect_stdout(out):
                ctrl = input.controls()
                regress().summary(
                    ssq="ssq" in ctrl, vif="vif" in ctrl, name=input.datasets()
                )
            return out.getvalue()

        @output(id="show_summary_code")
        @render.text
        def show_summary_code():
            rvar = input.rvar()
            evar = list(input.evar())
            ctrl = input.controls()
            _, fname, code = load_data()
            cmd = f"""{code}\nreg = rsm.regress(dataset={fname}, rvar="{rvar}", evar={evar})\nreg.summary(ssq={"ssq" in ctrl}, vif={"vif" in ctrl})"""
            return code_formatter(cmd)

        @output(id="regress_predict")
        @render.text
        def regress_predict():
            return ui.HTML(
                regress()
                .predict(df=self.datasets[input.pred_datasets()], ci=True)
                .round(3)
                .head()
                .to_html(classes="table table-striped data_preview", index=False)
            )

        @output(id="show_predict_code")
        @render.text
        def show_predict_code():
            rvar = input.rvar()
            evar = list(input.evar())
            _, fname, code = load_data()
            cmd = f"""{code}\nreg = rsm.regress(dataset={fname}, rvar="{rvar}", evar={evar})\nreg.predict(df={input.pred_datasets()}, ci=True)"""
            return code_formatter(cmd)

        @output(id="regress_plot")
        @render.plot()
        def regress_plot():
            return regress().plot(plots=input.regress_plots())

        @output(id="show_plot_code")
        @render.text
        def show_plot_code():
            rvar = input.rvar()
            evar = list(input.evar())
            plot_type = input.regress_plots()
            if plot_type != "None":
                _, fname, code = load_data()
                cmd = f"""{code}\nreg = rsm.regress(dataset={fname}, rvar="{rvar}", evar={evar})\nreg.plot("{plot_type}")"""
                return code_formatter(cmd)

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
# rsm.load_data(pkg="data", name="diamonds", dct=globals())
# mr = model_regress({"diamonds": diamonds, "diamonds100": diamonds.sample(100)})
# app = App(mr.shiny_ui(), mr.shiny_server)


def regress(
    data_dct: dict, host: str = "0.0.0.0", port: int = 8000, log_level: str = "warning"
):
    """
    Launch a Shiny-for-Python app for regression analysis
    """
    mr = model_regress(data_dct)
    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(
        App(mr.shiny_ui(), mr.shiny_server),
        host=host,
        port=port,
        log_level=log_level,
    )
