from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import webbrowser, nest_asyncio, uvicorn
import io
import pyrsm as rsm
from contextlib import redirect_stdout, redirect_stderr
import pyrsm.radiant.utils as ru
import pyrsm.radiant.model_utils as mu


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
            ),
        ),
    )


choices = {
    # "None": "None",
    "hist": "Histogram",
    # "sim": "Simulate",
}


class basics_single_mean:
    def __init__(self, datasets: dict, descriptions=None, open=True) -> None:
        ru.init(self, datasets, descriptions=descriptions, open=open)

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
                        ru.ui_plot(choices),
                    ),
                    ui.column(8, ru.ui_main_basics()),
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
        get_data, stop_app = ru.standard_reactives(self, input, session)
        ru.make_data_outputs(self, input, output)

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

            args = {
                "data": f"""{{"{data_name}": {data_name}}}""",
                "var": input.var(),
                "alt_hyp": input.alt_hyp(),
                "conf": input.conf(),
                "comp_value": input.comp_value(),
            }

            args_string = ru.drop_default_args(args, rsm.basics.single_mean)
            return f"""rsm.basics.single_mean({args_string})""", code

        show_code, estimate_new = mu.make_estimate(
            self,
            input,
            output,
            get_data,
            fun="basics.single_mean",
            ret="sm",
            ec=estimation_code,
            debug=True,
        )

        # def show_code():
        #     sc = estimation_code()
        #     return f"""{sc[1]}\nsm = {sc[0]}"""

        @reactive.Calc
        def estimate():
            locals()[input.datasets()] = self.datasets[
                input.datasets()
            ]  # get data into local scope
            return eval(estimation_code()[0])

        def summary_code():
            return f"""sm.summary()"""

        mu.make_summary(
            self,
            input,
            output,
            show_code,
            estimate,
            ret="sm",
            sum_fun=rsm.basics.single_mean.summary,
            sc=summary_code,
        )

        mu.make_plot(
            self,
            input,
            output,
            show_code,
            estimate,
            ret="sm",
        )


def single_mean(
    data_dct: dict,
    descriptions_dct: dict = None,
    open: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Radiant-for-Python app for single_mean hypothesis testing
    """
    rc = basics_single_mean(data_dct, descriptions_dct, open)
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

    demand_uk, demand_uk_description = rsm.load_data(pkg="basics", name="demand_uk")
    single_mean(
        {"demand_uk": demand_uk}, {"demand_uk": demand_uk_description}, open=True
    )
