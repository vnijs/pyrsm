from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import webbrowser, nest_asyncio, uvicorn
import io, os, signal
import pyrsm as rsm
from contextlib import redirect_stdout
from faicons import icon_svg
from ast import literal_eval
import pyrsm.radiant.utils as ru


def ui_summary():
    return ui.panel_conditional(
        "input.tabs == 'Summary'",
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
        ),
    )


def ui_predict(self):
    return ui.panel_conditional(
        "input.tabs == 'Predict'",
        ui.panel_well(
            ui.input_select(
                "pred_type",
                "Prediction input type:",
                ["None", "Data", "Command", "Data & Command"],
            ),
            ui.panel_conditional(
                "input.pred_type == 'Data' || input.pred_type == 'Data & Command'",
                ui.input_select("pred_datasets", "Prediction data:", self.dataset_list),
            ),
            ui.panel_conditional(
                "input.pred_type == 'Command' || input.pred_type == 'Data & Command'",
                ru.input_return_text_area(
                    "pred_cmd",
                    "Prediction command:",
                    rows=4,
                    placeholder="Specify a dictionary of values to to use for prediction, e.g., {'carat': 1, 'cut': 'Ideal'}",
                ),
            ),
            ui.input_checkbox("pred_ci", "Show pred. intervals"),
            ui.panel_conditional(
                "input.pred_ci == true",
                ui.input_slider(
                    id="conf",
                    label="Confidence level:",
                    min=0,
                    max=1,
                    value=0.95,
                ),
            ),
        ),
    )


def ui_plot():
    return ui.panel_conditional(
        "input.tabs == 'Plot'",
        ui.panel_well(
            ui.input_select(
                id="plots",
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
                "input.plots == 'corr'",
                ui.input_select(
                    "nobs",
                    "Number of data points plotted:",
                    {1000: "1,000", -1: "All"},
                ),
            ),
            ui.panel_conditional(
                "input.plots == 'pred'",
                ui.output_ui("ui_incl_evar"),
                ui.output_ui("ui_incl_interactions"),
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
            "Predict",
            ui.output_ui("show_predict_code"),
            ui.output_data_frame("predict"),
        ),
        ui.nav(
            "Plot",
            ui.output_ui("show_plot_code"),
            ui.output_plot("plot", height="800px", width="600px"),
        ),
        id="tabs",
    )


class model_regress:
    def __init__(self, datasets: dict, descriptions=None) -> None:
        ru.init(self, datasets, descriptions=descriptions)

    def shiny_ui(self):
        return ui.page_navbar(
            ru.head_content(),
            ui.nav(
                "Model > Linear regression (OLS)",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ui_summary(),
                        ui_predict(self),
                        ui_plot(),
                    ),
                    ui.column(8, ui_main()),
                ),
            ),
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/model-linear-regression.ipynb",
                "Linear regression (OLS) example notebook",
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
        @output(id="ui_rvar")
        @render.ui
        def ui_rvar():
            isNum = get_data()["var_types"]["isNum"]
            return ui.input_select(
                id="rvar",
                label="Response Variable",
                selected=None,
                choices=isNum,
            )

        @output(id="ui_evar")
        @render.ui
        def ui_evar():
            vars = get_data()["var_types"]["all"]
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
                    list(input.evar()), list(get_data()["var_types"]["isNum"].keys())
                )
                if len(isNum) > 0:
                    choices += ru.qterms(isNum, nway=int(input.show_interactions()[0]))
                choices += ru.iterms(input.evar(), nway=nway)
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
                choices = ru.iterms(input.evar(), nway=nway)
                return ui.input_select(
                    id="incl_interactions",
                    label="Interactions to include:",
                    selected=None,
                    choices=choices,
                    multiple=True,
                    size=min(8, len(choices)),
                    selectize=False,
                )

        def estimation_code():
            rvar = input.rvar()
            evar = list(input.evar())
            data_name, code = (get_data()[k] for k in ["data_name", "code"])
            if int(input.show_interactions()) > 0 and len(input.interactions()) > 0:
                return (
                    f"""rsm.regress(data={{"{data_name}": {data_name}}}, rvar="{rvar}", evar={evar}, int={list(input.interactions())})""",
                    code,
                )
            else:
                return (
                    f"""rsm.regress(data={{"{data_name}": {data_name}}}, rvar="{rvar}", evar={evar})""",
                    code,
                )

        def show_code():
            mc = estimation_code()
            return f"""{mc[1]}\nreg = {mc[0]}"""

        @reactive.Calc
        @reactive.event(input.run, ignore_none=True)
        def regress():
            locals()[input.datasets()] = self.datasets[
                input.datasets()
            ]  # get data into local scope
            return eval(estimation_code()[0])

        def summary_code():
            args = {c: True for c in input.controls()}
            if input.evar_test() is not None and len(input.evar_test()) > 0:
                args["test"] = list(input.evar_test())

            args_string = ru.drop_default_args(args, rsm.regress.summary)
            return f"""reg.summary({args_string})"""

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
                reg = regress()  # get model object into local scope
                cmd = summary_code()
                eval(cmd)
            return out.getvalue()

        def predict_code():
            args = {}
            try:
                # convert string to dict
                cmd = literal_eval(input.pred_cmd())
            except:
                cmd = None

            if input.pred_type() == "Data":
                args["df"] = input.pred_datasets()
            elif input.pred_type() == "Command" and not ru.is_empty(cmd):
                args["df"] = None
                args["cmd"] = cmd
            elif input.pred_type() == "Data & Command" and not ru.is_empty(cmd):
                args["df"] = input.pred_datasets()
                args["cmd"] = cmd
                args["dc"] = True

            ci = input.pred_ci()
            if ci:
                args.update({"ci": ci, "conf": input.conf()})

            args_string = ru.drop_default_args(args, rsm.regress.predict)
            return f"""reg.predict({args_string})"""

        @output(id="show_predict_code")
        @render.text
        def show_predict_code():
            return ru.code_formatter(
                f"""{show_code()}\npred = {predict_code()}""", self
            )

        @output(id="predict")
        @render.data_frame
        def predict():
            if input.pred_type() != "None":
                reg = regress()  # get model object into local scope
                if input.pred_type() in ["Data", "Data & Command"]:
                    locals()[input.pred_datasets()] = self.datasets[
                        input.pred_datasets()
                    ]  # get prediction data into local scope
                pred = eval(predict_code())

                summary = "Viewing rows {start} through {end} of {total}"
                if pred.shape[0] > 100_000:
                    pred = pred[:100_000]
                    summary += " (100K rows shown)"

                return render.DataTable(pred.round(3), summary=summary)
            else:
                return None

        def plot_code():
            plots = input.plots()
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
            plots = input.plots()
            if plots != "None":
                cmd = f"""{show_code()}\n{plot_code()}"""
                return ru.code_formatter(cmd, self)

        # functionality not yet available in shiny-for-python
        # def plot_height():
        #     plots = input.plots()
        #     if plots == "pred":
        #         return "800px"
        #     else:
        #         return "1000px"

        @output(id="plot")
        # @render.plot(height=plot_height)
        @render.plot
        def plot():
            plots = input.plots()
            if plots != "None":
                reg = regress()  # get model object into local scope
                cmd = f"""{plot_code()}"""
                return eval(cmd)

        @reactive.Effect
        def run_refresh():
            def update():
                with reactive.isolate():
                    if input.run() > 0:  # only update if run button was pressed
                        ui.update_action_button(
                            "run",
                            label="Re-estimate model",
                            icon=icon_svg("rotate"),
                        )

            if not ru.is_empty(input.evar()) and not ru.is_empty(input.rvar()):
                update()

            # not clear why this needs to be separate from the above
            if not ru.is_empty(input.interactions()):
                update()

        @reactive.Effect
        @reactive.event(input.run, ignore_none=True)
        def run_done():
            ui.update_action_button(
                "run",
                label="Estimate model",
                icon=icon_svg("play"),
            )


def regress(
    data_dct: dict,
    descriptions_dct: dict = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Radiant-for-Python app for linear regression analysis
    """
    rc = model_regress(data_dct, descriptions_dct)
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
