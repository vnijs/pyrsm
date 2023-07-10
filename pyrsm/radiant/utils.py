import black, os, signal, inspect
from htmltools import tags, div, css
from itertools import combinations
from shiny import render, ui, reactive
from faicons import icon_svg
from pathlib import Path
import pandas as pd
from pyrsm.utils import ifelse, md


def head_content():
    """
    Return the head content for the shiny app
    """
    return ui.head_content(
        ui.tags.script(
            (Path(__file__).parent / "www/returnTextAreaBinding.js").read_text()
        ),
        ui.tags.script((Path(__file__).parent / "www/copy.js").read_text()),
        # from https://github.com/rstudio/py-shiny/issues/491#issuecomment-1579138681
        ui.tags.link(
            href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/agate.min.css",
            rel="stylesheet",
        ),
        ui.tags.script(
            src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"
        ),
        ui.tags.style((Path(__file__).parent / "www/style.css").read_text()),
    )


def init(self, datasets, descriptions=None, open=True):
    """
    Initialize key 'self' values to be used in an app class
    """
    self.datasets = ifelse(isinstance(datasets, dict), datasets, {"dataset": datasets})
    if descriptions is None:
        self.descriptions = {
            k: "## No data description provided" for k in self.datasets.keys()
        }
    else:
        self.descriptions = ifelse(
            isinstance(descriptions, dict),
            descriptions,
            {"description": descriptions},
        )
    self.dataset_list = list(datasets.keys())
    self.open = open  # keep code windows open or closed by default
    self.stop_code = ""


def escape_quotes(cmd):
    return cmd.replace('"', '\\"').replace("'", "\\'")


def quote(v, k):
    if isinstance(v, str) and k not in ["data", "df"] and not (v[0] + v[-1] == "{}"):
        v = escape_quotes(v)
        return f'"{v}"'
    else:
        return v


def copy_icon(cmd):
    cmd = escape_quotes(cmd).replace("\n", "\\n")
    return (
        ui.input_action_link(
            "copy",
            None,
            icon=icon_svg("copy", width="1.2em", height="1.2em"),
            title="Copy to clipboard",
            onclick=f'copyToClipboard("{cmd}");',
        ),
    )


def code_formatter(code, self):
    """
    Format python code using black
    """
    cmd = self.stop_code = black.format_str(code, mode=black.Mode())
    return ui.TagList(
        ui.HTML(
            f"<details {ifelse(self.open, 'open', '')}><summary>View generated python code</summary>"
        ),
        copy_icon(cmd),
        ui.markdown(f"""\n```python\n{cmd.rstrip()}\n```"""),
        ui.tags.script("hljs.highlightAll();"),
        ui.HTML("</details>"),
        ui.br(),
    )


def drop_default_args(args, func):
    """
    Take a dictionary of arguments for a function and compare
    to the default arguments for that function
    Return a comma separate string of key-value pairs for
    non-default values where the value is quoted if it is a string
    """
    sig = inspect.signature(func).parameters
    keep = {k: args[k] for k, v in sig.items() if k in args and args[k] != v.default}

    return ", ".join(f"{k}={quote(v, k)}" for k, v in keep.items())


def get_data(self, input):
    """
    Set up data access for the app
    """
    data_name = input.datasets()
    data = self.datasets[data_name]
    description = self.descriptions[data_name]
    code = (
        f"import pyrsm as rsm\n# {data_name} = pd.read_parquet('{data_name}.parquet')"
    )
    if input.show_filter():
        code_sf = ""
        if not is_empty(input.data_filter()):
            code_sf += f""".query("{escape_quotes(input.data_filter())}")"""
        if not is_empty(input.data_sort()):
            code_sf += f""".sort_values({input.data_sort()})"""
        if not is_empty(input.data_slice()):
            code_sf += f""".iloc[{input.data_slice()}, :]"""

        if not is_empty(code_sf):
            data = eval(f"""data{code_sf}""")
            code = f"""{code}\n{data_name} = {data_name}{code_sf}"""

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

    return {
        "data": data,
        "data_name": data_name,
        "description": description,
        "var_types": var_types,
        "code": code,
    }


def standard_reactives(self, input, session):
    @reactive.Calc
    def rget_data():
        return get_data(self, input)

    @reactive.Effect
    @reactive.event(input.stop, ignore_none=True)
    async def stop_app():
        md(f"```python\n{input.stop_code}\n```")
        await session.app.stop()
        os.kill(os.getpid(), signal.SIGTERM)

    return rget_data, stop_app


def make_data_outputs(self, input, output):
    @output(id="show_data_code")
    @render.ui
    def show_data_code():
        return code_formatter(get_data(self, input)["code"], self)

    @output(id="show_data")
    @render.data_frame
    def show_data():
        data = get_data(self, input)["data"]
        summary = "Viewing rows {start} through {end} of {total}"
        if data.shape[0] > 100_000:
            data = data[:100_000]
            summary += " (100K rows shown)"

        return render.DataTable(data, summary=summary)

    @output(id="show_description")
    @render.ui
    def show_description():
        return ui.markdown(get_data(self, input)["description"])


def input_return_text_area(id, label, value="", rows=1, placeholder=""):
    classes = ["form-control", "returnTextArea"]
    area = tags.textarea(
        value,
        id=id,
        class_=" ".join(classes),
        style=css(width="100%", height=None, resize="vertical"),
        placeholder=placeholder,
        rows=rows,
        cols=None,
        autocomplete=False,
        spellcheck=False,
    )

    def shiny_input_label(id, label=None):
        cls = "control-label" + ("" if label else " shiny-label-null")
        return tags.label(label, class_=cls, id=id + "-label", for_=id)

    return div(
        shiny_input_label(id, label),
        area,
        None,
        class_="form-group shiny-input-container",
        style=css(width="100%"),
    )


def is_empty(x):
    return x is None or all(c.isspace() for c in x)


def qterms(vars, nway=2):
    return [f"I({v} ** {p})" for p in range(2, nway + 1) for v in vars]


def iterms(vars, nway=2, sep=":"):
    cvars = list(combinations(vars, 2))
    if nway > 2:
        cvars += list(combinations(vars, nway))
    return [f"{sep}".join(c) for c in cvars]


def ui_data(self):
    return (
        ui.panel_conditional(
            "input.tabs == 'Data'",
            ui.panel_well(
                ui.input_select("datasets", "Datasets:", self.dataset_list),
                ui.input_checkbox("show_filter", "Show data filter"),
                ui.panel_conditional(
                    "input.show_filter == true",
                    # ui.input_radio_buttons(
                    #     "data_language",
                    #     "Data language",
                    #     ["Pandas", "Polars", "SQL"],
                    #     inline=True,
                    # ),
                    input_return_text_area(
                        "data_filter",
                        "Data Filter:",
                        rows=2,
                        placeholder="Provide a filter (e.g., price >  5000) and press return",
                    ),
                    input_return_text_area(
                        "data_sort",
                        "Data sort:",
                        rows=2,
                        placeholder="Sort (e.g., ['color', 'price'], ascending=[True, False])) and press return",
                    ),
                    input_return_text_area(
                        "data_slice",
                        "Data slice (rows):",
                        rows=1,
                        placeholder="e.g., 0:50 and press return",
                    ),
                ),
            ),
        ),
    )


def ui_summary(*args):
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
            ui.output_ui("ui_rvar"),
            ui.output_ui("ui_lev"),
            ui.output_ui("ui_evar"),
            *args,
        ),
    )


def ui_plot(choices, *args):
    return ui.panel_conditional(
        "input.tabs == 'Plot'",
        ui.panel_well(
            ui.input_select(
                id="plots",
                label="Plots",
                selected=None,
                choices=choices,
            ),
            *args,
        ),
    )


def ui_data_main():
    data_main = ui.nav(
        "Data",
        ui.output_ui("show_data_code"),
        ui.output_data_frame("show_data"),
        ui.output_ui("show_description"),
    )
    return data_main


def ui_main_basics():
    return ui.navset_tab_card(
        ui_data_main(),
        ui.nav(
            "Summary",
            ui.output_ui("show_summary_code"),
            ui.output_text_verbatim("summary"),
        ),
        ui.nav(
            "Plot",
            ui.output_ui("show_plot_code"),
            ui.output_plot("plot", height="500px", width="700px"),
        ),
        id="tabs",
    )


def ui_main_model():
    return ui.navset_tab_card(
        ui_data_main(),
        ui.nav(
            "Summary",
            ui.output_ui("show_estimation_code"),
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
            ui.output_plot("plot", height="800px", width="700px"),
        ),
        id="tabs",
    )


def ui_stop():
    return (
        ui.nav_control(
            ui.input_action_link(
                "stop", "Stop", icon=icon_svg("stop"), onclick="window.close();"
            )
        ),
    )


def ui_help(link, example):
    return (
        ui.nav_menu(
            "Help",
            ui.nav_control(
                ui.a(
                    icon_svg("question"),
                    example,
                    href=link,
                    target="_blank",
                ),
            ),
            ui.nav_control(
                ui.a(
                    icon_svg("github"),
                    "Radiant-for-python source code",
                    href="https://github.com/vnijs/pyrsm/tree/main/pyrsm/radiant",
                    target="_blank",
                ),
            ),
            ui.nav_control(
                ui.a(
                    icon_svg("github"),
                    "Pyrsm source code",
                    href="https://github.com/vnijs/pyrsm/tree/main",
                    target="_blank",
                ),
            ),
            ui.nav_control(
                ui.a(
                    icon_svg("docker"),
                    "Rady MSBA docker container",
                    href="https://github.com/radiant-rstats/docker",
                    target="_blank",
                ),
            ),
            align="right",
        ),
    )


def reestimate(input):
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

        if not is_empty(input.evar()) and not is_empty(input.rvar()):
            update()

        # not clear why this needs to be separate from the above
        if not is_empty(input.interactions()):
            update()

    @reactive.Effect
    @reactive.event(input.run, ignore_none=True)
    def run_done():
        ui.update_action_button(
            "run",
            label="Estimate model",
            icon=icon_svg("play"),
        )

    return run_refresh, run_done


# getting data to work as a separate nav item caused problems
# def ui_data(self):
#     return ui.nav(
#         "Data",
#         ui.row(
#             ui.column(
#                 3,
#                 ui.panel_well(
#                     ui.input_select("datasets", "Datasets:", self.dataset_list)
#                 ),
#                 ui.panel_well(
#                     ui.input_checkbox("show_filter", "Show data filter", value=True),
#                     ui.panel_conditional(
#                         "input.show_filter == true",
#                         ui.input_radio_buttons(
#                             "data_language",
#                             "Data language",
#                             choices=["Pandas", "Polars", "SQL"],
#                             inline=True,
#                         ),
#                         input_return_text_area(
#                             "data_filter",
#                             "Data Filter:",
#                             rows=2,
#                             placeholder="Provide a filter (e.g., price >  5000) and press return",
#                         ),
#                         input_return_text_area(
#                             "data_sort",
#                             "Data sort:",
#                             rows=2,
#                             placeholder="Sort (e.g., ['color', 'price'], ascending=[True, False])) and press return",
#                         ),
#                         input_return_text_area(
#                             "data_slice",
#                             "Data slice (rows):",
#                             rows=1,
#                             placeholder="e.g., 1:50 and press return",
#                         ),
#                     ),
#                 ),
#             ),
#             ui.column(
#                 8,
#                 ui.output_ui("show_data_code"),
#                 ui.output_data_frame("show_data"),
#                 ui.output_ui("show_description"),
#             ),
#         ),
#     )
