import io
import matplotlib.pyplot as plt
from shiny import render, ui, reactive, req
from contextlib import redirect_stdout, redirect_stderr
import pyrsm.radiant.utils as ru
from pyrsm.utils import intersect
import numpy as np
import pandas as pd
import pyrsm as rsm


def ui_predict(self, show_ci=True):
    if show_ci:
        ci = (
            ui.input_checkbox(
                "pred_ci",
                "Show conf. intervals",
                value=self.state.get("pred_ci", False),
            ),
            ui.panel_conditional(
                "input.pred_ci == true",
                ui.input_slider(
                    id="conf",
                    label="Confidence level:",
                    min=0,
                    max=1,
                    value=self.state.get("conf", 0.95),
                ),
            ),
        )
    else:
        ci = None

    return ui.panel_conditional(
        "input.tabs == 'Predict'",
        ui.panel_well(
            ui.input_select(
                "pred_type",
                "Prediction input type:",
                ["None", "Data", "Command", "Data & Command"],
                selected=self.state.get("pred_type", "None"),
            ),
            ui.panel_conditional(
                "input.pred_type == 'Data' || input.pred_type == 'Data & Command'",
                ui.input_select(
                    "pred_datasets",
                    "Prediction data:",
                    self.dataset_list,
                    selected=self.state.get("pred_data", None),
                ),
            ),
            ui.panel_conditional(
                "input.pred_type == 'Command' || input.pred_type == 'Data & Command'",
                ru.input_return_text_area(
                    "pred_cmd",
                    "Prediction command:",
                    rows=3,
                    value=self.state.get("pred_cmd", ""),
                    placeholder="Specify a dictionary of values to to use for prediction, e.g., {'carat': 1, 'cut': 'Ideal'}",
                ),
            ),
            ci,
        ),
    )


def make_model_inputs(self, input, output, get_data, mod_type="isBin"):
    if isinstance(mod_type, str):

        @output(id="ui_rvar")
        @render.ui
        def ui_rvar():
            isType = get_data()["var_types"][mod_type]
            return ui.input_select(
                id="rvar",
                label="Response Variable:",
                selected=self.state.get("rvar", None),
                choices=isType,
            )

    elif isinstance(mod_type, dict):

        @output(id="ui_rvar")
        @render.ui
        def ui_rvar():
            isType = get_data()["var_types"][mod_type[input.mod_type()]]
            return ui.input_select(
                id="rvar",
                label="Response Variable:",
                selected=self.state.get("rvar", None),
                choices=isType,
            )

    @output(id="ui_evar")
    @render.ui
    def ui_evar():
        vars = get_data()["var_types"]["all"].copy()
        if (input.rvar() is not None) and (input.rvar() in vars):
            del vars[input.rvar()]

        return ui.input_select(
            id="evar",
            label="Explanatory Variables:",
            selected=self.state.get("evar", []),
            choices=vars,
            multiple=True,
            size=min(8, len(vars)),
            selectize=False,
        )


def make_int_inputs(self, input, output, get_data):
    @output(id="ui_interactions")
    @render.ui
    def ui_interactions():
        if len(input.evar()) > 1 and input.show_interactions() is not None:
            choices = []
            nway = int(input.show_interactions())
            isNum = intersect(
                list(input.evar()), list(get_data()["var_types"]["isNum"].keys())
            )
            if len(isNum) > 0:
                # choices += ru.qterms(isNum, nway=int(input.show_interactions()[0]))
                choices += ru.qterms(isNum, nway=nway)
            choices += ru.iterms(input.evar(), nway=nway)
            return ui.input_select(
                id="interactions",
                label=None,
                selected=self.state.get("interactions", None),
                choices=choices,
                multiple=True,
                size=min(8, len(choices)),
                selectize=False,
            )

    @output(id="ui_evar_test")
    @render.ui
    def ui_evar_test():
        choices = {e: e for e in input.evar()}
        if (
            "show_interactions" in input
            and input.show_interactions() is not None
            and int(input.show_interactions()[0]) > 1
        ):
            choices.update({e: e for e in input.interactions()})
        return ui.input_selectize(
            id="evar_test",
            label="Variables to test:",
            selected=self.state.get("evar_test", None),
            choices=choices,
            multiple=True,
        )

    @output(id="ui_incl_evar")
    @render.ui
    def ui_incl_evar():
        if len(input.evar()) > 0:
            return ui.input_select(
                id="incl_evar",
                label="Explanatory variables to include:",
                selected=self.state.get("incl_evar", None),
                choices=input.evar(),
                multiple=True,
                size=min(8, len(input.evar())),
                selectize=False,
            )

    @output(id="ui_incl_interactions")
    @render.ui
    def ui_incl_interactions():
        if len(input.evar()) > 1:
            if "show_interactions" in input and input.show_interactions() is not None:
                nway = int(input.show_interactions())
            else:
                nway = 2
            choices = ru.iterms(input.evar(), nway=nway)
            return ui.input_select(
                id="incl_interactions",
                label="Interactions to include:",
                selected=self.state.get("incl_interactions", None),
                choices=choices,
                multiple=True,
                size=min(8, len(choices)),
                selectize=False,
            )


def make_estimate(
    self,
    input,
    output,
    get_data,
    fun,
    ret,
    module="model.",
    run=True,
    ec=None,
    debug=False,
):
    if ec is None:

        def estimation_code():
            data_name, code = (get_data()[k] for k in ["data_name", "code"])
            # from https://discord.com/channels/1109483223987277844/1127817202804985917/1129530385429176371
            # does not work with ifelse for some reason
            # inp = ifelse("lev" in input, input.lev(), None)
            if "lev" in input:
                inp = input.lev()
            else:
                inp = None
            if "weights" in input:
                weights = input.weights()
                if weights == "None":
                    weights = None
            else:
                weights = None

            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "rvar": input.rvar(),
                "lev": inp,
                "evar": list(input.evar()),
                "weights": weights,
            }

            if (
                "show_interactions" in input
                and input.show_interactions() is not None
                and int(input.show_interactions()) > 0
                and len(input.interactions()) > 0
            ):
                args["ivar"] = list(input.interactions())

            if ret == "rf":
                if input.max_features() in ["sqrt", "log2"]:
                    max_features = input.max_features()
                else:
                    max_features = int(input.max_features())

                args.update(
                    {
                        "mod_type": input.mod_type(),
                        "n_estimators": input.n_estimators(),
                        "max_features": max_features,
                        "min_samples_leaf": input.min_samples_leaf(),
                        "max_samples": input.max_samples(),
                        "random_state": input.random_state(),
                    }
                )
            elif ret == "nn":
                args.update(
                    {
                        "mod_type": input.mod_type(),
                        "hidden_layer_sizes": eval(input.hidden_layer_sizes()),
                        "activation": input.activation(),
                        "solver": input.solver(),
                        "alpha": input.alpha(),
                        # "batch_size": input.batch_size(),
                        "learning_rate_init": input.learning_rate_init(),
                        "max_iter": input.max_iter(),
                    }
                )

            args_str = ru.drop_default_args(args, getattr(rsm, fun))

            if "extra_args" in input and input.extra_args() != "":
                args_str += ", " + input.extra_args()
            return f"""rsm.{module}{fun}({args_str})""", code

    else:
        estimation_code = ec

    if debug:

        @output(id="show_estimation_code")
        @render.text
        def show_estimation_code():
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                eval(estimation_code()[0])

            return out.getvalue()

    def show_code():
        sc = estimation_code()
        return f"""{sc[1]}\n{ret} = {sc[0]}"""

    if run:

        @reactive.Calc
        @reactive.event(input.run, ignore_none=True)
        def estimate():
            locals()[input.datasets()] = self.datasets[
                input.datasets()
            ]  # get data into local scope

            ec = estimation_code()
            # exec for multi-line
            # https://stackoverflow.com/a/12698067/1974918
            exec(ec[1])
            return eval(ec[0])

    else:

        @reactive.Calc
        def estimate():
            locals()[input.datasets()] = self.datasets[
                input.datasets()
            ]  # get data into local scope

            ec = estimation_code()
            exec(ec[1])
            return eval(ec[0])

    return show_code, estimate


def make_summary(
    self, input, output, session, show_code, estimate, ret, sum_fun, sc=None
):
    if sc is None:

        def summary_code():
            if "controls" in input:
                args = {c: True for c in input.controls()}
            else:
                args = {}
            if (
                "evar_test" in input
                and input.evar_test() is not None
                and len(input.evar_test()) > 0
            ):
                args["test"] = list(input.evar_test())

            args_string = ru.drop_default_args(args, sum_fun)
            return f"""{ret}.summary({args_string})"""

    else:
        summary_code = sc

    @output(id="show_summary_code")
    @render.text
    def show_summary_code():
        cmd = f"""{show_code()}\n{summary_code()}"""
        return ru.code_formatter(cmd, self, input, session, id="copy_summary")

    @output(id="summary")
    @render.text
    def summary():
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            locals()[ret] = estimate()  # get model object into local scope
            eval(summary_code())
        return out.getvalue()


def make_predict(
    self, input, output, session, show_code, estimate, ret, pred_fun, show_ci=False
):
    def predict_code():
        args = {}
        cmd = input.pred_cmd().strip()

        if input.pred_type() == "Data":
            args["data"] = input.pred_datasets()
        elif input.pred_type() == "Command" and not ru.is_empty(cmd):
            args["data"] = None
            args["cmd"] = cmd
        elif input.pred_type() == "Data & Command" and not ru.is_empty(cmd):
            args["data"] = input.pred_datasets()
            args["data_cmd"] = cmd

        if "pred_ci" in input:
            ci = input.pred_ci()
            if ci:
                args.update({"ci": ci, "conf": input.conf()})

        args_string = ru.drop_default_args(
            args, pred_fun, ignore=["data", "cmd", "data_cmd"]
        )
        return f"""{ret}.predict({args_string})"""

    @output(id="show_predict_code")
    @render.text
    def show_predict_code():
        return ru.code_formatter(
            f"""{show_code()}\npred = {predict_code()}""",
            self,
            input,
            session,
            id="copy_predict",
        )

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


def make_plot(self, input, output, session, show_code, estimate, ret, pc=None):
    if pc is None:

        def plot_code():
            plots = input.plots()
            if plots == "pred":
                incl = list(input.incl_evar())
                if "incl_interactions" in input:
                    incl_int = list(input.incl_interactions())
                else:
                    incl_int = []
                cmd = f""", incl={incl}"""
                if len(incl_int) > 0:
                    cmd += f""", incl_int={incl_int}"""
                cmd = f"""{ret}.plot(plots="{plots}" {cmd})"""
            elif plots == "corr":
                cmd = f"""{ret}.plot(plots="{plots}", nobs={input.nobs()})"""
            else:
                cmd = f"""{ret}.plot(plots="{plots}")"""
            return cmd

    else:
        plot_code = pc

    @output(id="show_plot_code")
    @render.text
    def show_plot_code():
        plots = input.plots()
        if plots != "None":
            cmd = f"""{show_code()}\n{plot_code()}"""
            return ru.code_formatter(cmd, self, input, session, id="copy_plot")

    @reactive.Calc
    def gen_plot():
        locals()[ret] = estimate()
        eval(f"""{plot_code()}""")
        fig = plt.gcf()
        width, height = fig.get_size_inches()  # Get the size in inches
        return fig, width * 96, height * 96

    @output(id="plot")
    @render.plot
    def plot():
        plots = input.plots()
        if plots != "None":
            return gen_plot()[0]

    @output(id="plot_container")
    @render.ui
    def plot_container():
        req(estimate(), input.plots())
        width, height = gen_plot()[1:]
        return ui.output_plot("plot", height=f"{height}px", width=f"{width}px")
