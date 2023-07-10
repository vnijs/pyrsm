import io
from shiny import render, ui, reactive
from contextlib import redirect_stdout, redirect_stderr
import pyrsm.radiant.utils as ru
from ..utils import intersect, ifelse
import numpy as np
import pandas as pd
import pyrsm as rsm


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
                    rows=3,
                    placeholder="Specify a dictionary of values to to use for prediction, e.g., {'carat': 1, 'cut': 'Ideal'}",
                ),
            ),
            ui.input_checkbox("pred_ci", "Show conf. intervals"),
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


def make_model_inputs(input, output, get_data, type):
    @output(id="ui_rvar")
    @render.ui
    def ui_rvar():
        isType = get_data()["var_types"][type]
        return ui.input_select(
            id="rvar",
            label="Response Variable:",
            selected=None,
            choices=isType,
        )

    @output(id="ui_evar")
    @render.ui
    def ui_evar():
        vars = get_data()["var_types"]["all"]
        if (input.rvar() is not None) and (input.rvar() in vars):
            del vars[input.rvar()]

        return ui.input_select(
            id="evar",
            label="Explanatory Variables:",
            selected=None,
            choices=vars,
            multiple=True,
            size=min(8, len(vars)),
            selectize=False,
        )


def make_int_inputs(input, output, get_data):
    @output(id="ui_interactions")
    @render.ui
    def ui_interactions():
        if len(input.evar()) > 1:
            choices = []
            nway = int(input.show_interactions())
            isNum = intersect(
                list(input.evar()), list(get_data()["var_types"]["isNum"].keys())
            )
            if len(isNum) > 0:
                choices += ru.qterms(isNum, nway=int(input.show_interactions()[0]))
            choices += ru.iterms(input.evar(), nway=nway)
            return ui.input_select(
                id="interactions",
                label=None,
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


def new_estimation_code():
    data_name, code = (get_data()[k] for k in ["data_name", "code"])
    args = {
        "data": {f"{data_name}": data_name},
        "rvar": input.rvar(),
        "lev": input.lev(),
        "evar": list(input.evar()),
        "int": list(input.interactions()),
    }

    if int(input.show_interactions()) > 0 and len(input.interactions()) > 0:
        args["int"] = None

    args_str = ", ".join(f"{k}={ru.quote(v)}" for k, v in args.items() if v is not None)
    return f"""rsm.{fun}({args_str})""", code


def make_estimate(self, input, output, get_data, fun, ret, debug=False):
    def estimation_code():
        data_name, code = (get_data()[k] for k in ["data_name", "code"])
        try:
            inp = input.lev()
        except:
            inp = None

        args = {
            "data": f"""{{"{data_name}": {data_name}}}""",
            "rvar": input.rvar(),
            "lev": inp,
            "evar": list(input.evar()),
        }

        if int(input.show_interactions()) > 0 and len(input.interactions()) > 0:
            args["ivar"] = list(input.interactions())

        args_str = ", ".join(
            f"{k}={ru.quote(v, k)}" for k, v in args.items() if v is not None
        )
        return f"""rsm.{fun}({args_str})""", code

    if debug:

        @output(id="show_estimation_code")
        @render.text
        def show_estimation_code():
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                try:
                    print(input.x())  # why is there no error printed anywhere?
                except Exception as err:
                    print(err)  # why is there no error printed anywhere?

                print(estimation_code())
            return out.getvalue()

    def show_code():
        sc = estimation_code()
        return f"""{sc[1]}\n{ret} = {sc[0]}"""

    @reactive.Calc
    @reactive.event(input.run, ignore_none=True)
    def estimate():
        locals()[input.datasets()] = self.datasets[
            input.datasets()
        ]  # get data into local scope
        return eval(estimation_code()[0])

    return show_code, estimate


def make_summary(self, input, output, show_code, estimate, ret, sum_fun):
    def summary_code():
        args = {c: True for c in input.controls()}
        if input.evar_test() is not None and len(input.evar_test()) > 0:
            args["test"] = list(input.evar_test())

        args_string = ru.drop_default_args(args, sum_fun)
        return f"""{ret}.summary({args_string})"""
        # return f"""{ret}.summary()"""

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
            locals()[ret] = estimate()  # get model object into local scope
            eval(summary_code())
        return out.getvalue()


def make_predict(self, input, output, show_code, estimate, ret, pred_fun):
    def predict_code():
        args = {}
        cmd = input.pred_cmd().strip()

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

        args_string = ru.drop_default_args(args, pred_fun)
        return f"""{ret}.predict({args_string})"""

    @output(id="show_predict_code")
    @render.text
    def show_predict_code():
        return ru.code_formatter(f"""{show_code()}\npred = {predict_code()}""", self)

    @output(id="predict")
    @render.data_frame
    def predict():
        if input.pred_type() != "None":
            locals()[ret] = estimate()  # get model object into local scope
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


def make_plot(self, input, output, show_code, estimate, ret):
    def plot_code():
        plots = input.plots()
        if plots == "pred":
            incl = list(input.incl_evar())
            incl_int = list(input.incl_interactions())
            cmd = f""", incl={incl}"""
            if len(incl_int) > 0:
                cmd += f""", incl_int={incl_int}"""
            cmd = f"""{ret}.plot(plots="{plots}" {cmd})"""
        elif plots == "corr":
            cmd = f"""{ret}.plot(plots="{plots}", nobs={input.nobs()})"""
        else:
            cmd = f"""{ret}.plot(plots="{plots}")"""
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
            locals()[ret] = estimate()  # get model object into local scope
            return eval(f"""{plot_code()}""")
