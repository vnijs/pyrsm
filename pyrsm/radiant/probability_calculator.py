from shiny import App, render, ui, reactive, Inputs, Outputs, Session, req
import webbrowser
import nest_asyncio
import uvicorn
import signal
import os
import io
import sys
import tempfile
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr
import pyrsm as rsm
import pyrsm.radiant.utils as ru
import pyrsm.basics.probability_calculator as pc

pc_type = {"values": "Values", "probs": "Probabilities"}
pc_dist = {v: k for k, v in pc.pc_dist.items()}


def ui_type():
    return ui.panel_well(
        ui.input_radio_buttons(
            id="type",
            label="Input type:",
            selected="values",
            choices=pc_type,
            inline=True,
        ),
        ui.panel_conditional(
            "input.type == 'values'",
            ru.make_side_by_side(
                ui.input_numeric("lb", "Lower bound:", value=None),
                ui.input_numeric("ub", "Upper bound:", value=None),
            ),
        ),
        ui.panel_conditional(
            "input.type == 'probs'",
            ru.make_side_by_side(
                ui.input_numeric(
                    "plb", "Lower bound:", value=None, min=0, max=1, step=0.05
                ),
                ui.input_numeric(
                    "pub", "Upper bound:", value=None, min=0, max=1, step=0.05
                ),
            ),
        ),
        ui.input_numeric("dec", "Decimals:", value=3),
    )


def ui_summary():
    return ui.panel_well(
        ui.input_select(
            id="dist",
            label="Distribution:",
            selected="binom",
            choices=pc_dist,
        ),
        ui.panel_conditional("input.dist == 'binom'", ui.output_ui("ui_pc_binom")),
        ui.panel_conditional("input.dist == 'chisq'", ui.output_ui("ui_pc_chisq")),
        ui.panel_conditional("input.dist == 'disc'", ui.output_ui("ui_pc_disc")),
        ui.panel_conditional("input.dist == 'expo'", ui.output_ui("ui_pc_expo")),
        ui.panel_conditional("input.dist == 'fdist'", ui.output_ui("ui_pc_fdist")),
        ui.panel_conditional("input.dist == 'lnorm'", ui.output_ui("ui_pc_lnorm")),
        ui.panel_conditional("input.dist == 'norm'", ui.output_ui("ui_pc_norm")),
        ui.panel_conditional("input.dist == 'pois'", ui.output_ui("ui_pc_pois")),
        ui.panel_conditional("input.dist == 'tdist'", ui.output_ui("ui_pc_tdist")),
        ui.panel_conditional("input.dist == 'unif'", ui.output_ui("ui_pc_unif")),
    )


def ui_main_pc(height="500px", width="700px"):
    return ui.panel_well(
        ui.output_ui("show_summary_code"),
        ui.output_text_verbatim("summary"),
        ui.output_ui("plot_container"),
        id="pc_well",
    )


class basics_probability_calculator:
    def __init__(self, code=True) -> None:
        self.code = code

    def shiny_ui(self, *args):
        return ui.page_navbar(
            ru.head_content(),
            ui.nav(
                "<< Basics > Probability calculator >>",
                ui.row(
                    ui.column(
                        3,
                        ui_summary(),
                        ui_type(),
                    ),
                    ui.column(8, ui_main_pc()),
                ),
            ),
            *args,
            ru.ui_help(
                "https://github.com/vnijs/pyrsm/blob/main/examples/basics-probability-calculator.ipynb",
                "Probability calculator example notebook",
            ),
            ru.ui_stop(),
            title="Radiant for Python",
            inverse=True,
            id="navbar_id",
        )

    def shiny_server(self, input: Inputs, output: Outputs, session: Session):
        @output(id="ui_pc_binom")
        @render.ui
        def ui_pc_binom():
            return ru.make_side_by_side(
                ui.input_numeric("binom_n", label="n:", value=10, min=0),
                ui.input_numeric("binom_p", label="p:", value=0.2, min=0, max=1),
            )

        @output(id="ui_pc_chisq")
        @render.ui
        def ui_pc_chisq():
            return ui.input_numeric(
                "chisq_df", label="Degrees of freedom:", value=1, min=1
            )

        @output(id="ui_pc_disc")
        @render.ui
        def ui_pc_disc():
            return ui.panel_well(
                ru.input_return_text_area(
                    "disc_values",
                    label="Values:",
                    placeholder="Insert list [1, 3, 5]",
                    value="[1, 3, 5]",
                ),
                ru.input_return_text_area(
                    "disc_probs",
                    label="Probabilities:",
                    placeholder="Insert list [1/4, 1/8, 5/8]",
                    value="[1 / 4, 1 / 8, 5 / 8]",
                ),
            )

        @output(id="ui_pc_expo")
        @render.ui
        def ui_pc_expo():
            return ui.input_numeric("expo_rate", label="Rate:", value=1, min=1)

        @output(id="ui_pc_fdist")
        @render.ui
        def ui_pc_fdist():
            return ru.make_side_by_side(
                ui.input_numeric(
                    "fdist_df1", label="Degrees of freedom 1:", value=10, min=0
                ),
                ui.input_numeric(
                    "fdist_df2", label="Degrees of freedom 2:", value=10, min=0
                ),
            )

        @output(id="ui_pc_lnorm")
        @render.ui
        def ui_pc_lnorm():
            return ru.make_side_by_side(
                ui.input_numeric("lnorm_mean", label="Mean log:", value=0),
                ui.input_numeric("lnorm_stdev", label="St. dev. log:", value=1, min=0),
            )

        @output(id="ui_pc_norm")
        @render.ui
        def ui_pc_norm():
            return ru.make_side_by_side(
                ui.input_numeric("norm_mean", label="Mean:", value=0),
                ui.input_numeric("norm_stdev", label="St. dev.:", value=1, min=0),
            )

        @output(id="ui_pc_pois")
        @render.ui
        def ui_pc_pois():
            return ui.input_numeric(
                "pois_lambda", label="Lambda:", value=0.5, min=0, step=0.1
            )

        @output(id="ui_pc_tdist")
        @render.ui
        def ui_pc_tdist():
            return ui.input_numeric(
                "tdist_df", label="Degrees of freedom:", value=10, min=1
            )

        @output(id="ui_pc_unif")
        @render.ui
        def ui_pc_unif():
            return ru.make_side_by_side(
                ui.input_numeric("unif_min", label="Minimum:", value=0),
                ui.input_numeric("unif_max", label="Maximum:", value=1),
            )

        def estimation_code():
            dist = input.dist()
            ignore = [""]
            if dist == "binom":
                args = {
                    "n": input.binom_n(),
                    "p": input.binom_p(),
                }
            elif dist == "chisq":
                args = {"df": input.chisq_df()}
            elif dist == "disc":
                args = {
                    "v": input.disc_values(),
                    "p": input.disc_probs(),
                }
                ignore = ["v", "p"]
            elif dist == "expo":
                args = {"rate": input.expo_rate()}
            elif dist == "fdist":
                args = {
                    "df1": input.fdist_df1(),
                    "df2": input.fdist_df2(),
                }
            elif dist == "lnorm":
                args = {
                    "meanlog": input.lnorm_mean(),
                    "sdlog": input.lnorm_stdev(),
                }
            elif dist == "norm":
                args = {
                    "mean": input.norm_mean(),
                    "stdev": input.norm_stdev(),
                }
            elif dist == "pois":
                args = {"lamb": input.pois_lambda()}
            elif dist == "tdist":
                args = {"df": input.tdist_df()}
            elif dist == "unif":
                args = {"min": input.unif_min(), "max": input.unif_max()}
            else:
                pass

            # inputs in args must be available for calculation to work
            [req(v is not None) for v in args.values()]

            if input.type() == "values":
                args.update({"lb": input.lb(), "ub": input.ub()})
            else:
                args.update({"plb": input.plb(), "pub": input.pub()})

            fun = getattr(
                rsm.basics.probability_calculator_functions, f"prob_{input.dist()}"
            )
            args_string = ru.drop_default_args(args, fun, ignore=ignore)
            return f"""rsm.basics.prob_calc("{dist}", {args_string})"""

        @reactive.Calc
        def estimate():
            return eval(estimation_code())

        def summary_code():
            args = {"dec": input.dec()}
            fun = getattr(
                rsm.basics.probability_calculator_functions,
                f"summary_prob_{input.dist()}",
            )
            args_string = ru.drop_default_args(args, fun)
            return f"""pc.summary({args_string})"""

        @output(id="show_summary_code")
        @render.text
        def show_summary_code():
            cmd = f"""import pyrsm as rsm\npc = {estimation_code()}\n{summary_code()}\npc.plot()"""
            return ru.code_formatter(cmd, self, input, session)

        @output(id="summary")
        @render.text
        def summary():
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                pc = estimate()  # get estimation object into local scope
                eval(summary_code())
            return out.getvalue()

        @reactive.Calc
        def gen_plot():
            locals()["pc"] = estimate()
            eval("""pc.plot()""")
            fig = plt.gcf()
            width, height = fig.get_size_inches()  # Get the size in inches
            return fig, width * 96, height * 96

        @output(id="plot")
        @render.plot
        def plot():
            return gen_plot()[0]

        @output(id="plot_container")
        @render.ui
        def plot_container():
            req(estimate())
            width, height = gen_plot()[1:]
            return ui.output_plot("plot", height=f"{height}px", width=f"{width}px")

        # --- section standard for all apps ---
        # stops returning code if moved to utils
        @reactive.Effect
        @reactive.event(input.stop, ignore_none=True)
        async def stop_app():
            rsm.md(f"```python\n{self.stop_code}\n```")
            await session.app.stop()
            os.kill(os.getpid(), signal.SIGTERM)

            return self.stop_code


def prob_calc(
    code: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "critical",
    debug: bool = False,
):
    """
    Launch a Radiant-for-Python app for compare means hypothesis testing
    """
    rc = basics_probability_calculator(code=code)
    nest_asyncio.apply()
    webbrowser.open(f"http://{host}:{port}")
    print(f"Listening on http://{host}:{port}")
    ru.message()

    # redirect stdout and stderr to the temporary file
    if not debug:
        temp = tempfile.NamedTemporaryFile()
        sys.stdout = open(temp.name, "w")
        sys.stderr = open(temp.name, "w")

    uvicorn.run(
        App(rc.shiny_ui(), rc.shiny_server),
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    prob_calc(debug=False)
