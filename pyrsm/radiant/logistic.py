from shiny import App, ui, render, Inputs, Outputs, Session
import webbrowser, nest_asyncio, uvicorn
import pyrsm as rsm
from faicons import icon_svg
import pyrsm.radiant.utils as ru
import pyrsm.radiant.model_utils as mu

choices = {
    "None": "None",
    "dist": "Distribution",
    "corr": "Correlation",
    "pred": "Prediction plot",
    "vimp": "Permutation importance",
    "or": "OR plot",
}

controls = {
    "ci": "Confidence intervals",
    "vif": "VIF",
}

summary_extra = (
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
    ui.input_checkbox_group(
        "controls",
        "Additional output:",
        choices=controls,
    ),
    ui.output_ui("ui_evar_test"),
)

plots_extra = (
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
)


class model_logistic:
    def __init__(self, datasets: dict, descriptions=None, open=True) -> None:
        ru.init(self, datasets, descriptions=descriptions, open=open)

    def shiny_ui(self):
        return ui.page_navbar(
            ru.head_content(),
            ui.nav(
                "Model > Logistic regression (GLM)",
                ui.row(
                    ui.column(
                        3,
                        ru.ui_data(self),
                        ru.ui_summary(summary_extra),
                        mu.ui_predict(self),
                        ru.ui_plot(choices, plots_extra),
                    ),
                    ui.column(8, ru.ui_main_model()),
                ),
            ),
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/model-logistic-regression.ipynb",
                "Logistic regression (GLM) example notebook",
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

        # --- section standard for all model apps ---
        run_refresh, run_done = ru.reestimate(input)

        # --- section unique to each app ---
        mu.make_model_inputs(input, output, get_data, "isBin")

        @output(id="ui_lev")
        @render.ui
        def ui_lev():
            levs = list(get_data()["data"][input.rvar()].unique())
            return ui.input_select(
                id="lev",
                label="Choose level:",
                selected=levs[0],
                choices=levs,
            )

        mu.make_int_inputs(input, output, get_data)
        show_code, estimate = mu.make_estimate(
            self, input, output, get_data, fun="logistic", ret="lr", debug=True
        )
        mu.make_summary(
            self,
            input,
            output,
            show_code,
            estimate,
            ret="lr",
            sum_fun=rsm.logistic.summary,
        )
        mu.make_predict(
            self,
            input,
            output,
            show_code,
            estimate,
            ret="lr",
            pred_fun=rsm.logistic.predict,
        )
        mu.make_plot(
            self,
            input,
            output,
            show_code,
            estimate,
            ret="lr",
        )


def logistic(
    data_dct: dict,
    descriptions_dct: dict = None,
    open: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "warning",
):
    """
    Launch a Radiant-for-Python app for logistic regression analysis
    """
    rc = model_logistic(data_dct, descriptions_dct, open=open)
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


if __name__ == "__main__":
    import pyrsm as rsm

    titanic, titanic_description = rsm.load_data(pkg="data", name="titanic")
    logistic({"titanic": titanic}, {"titanic": titanic_description}, open=True)
