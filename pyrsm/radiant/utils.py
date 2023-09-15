import black
import inspect
from shiny import req
from htmltools import tags, div, css
from itertools import combinations
from shiny import render, ui, reactive
from faicons import icon_svg
import pandas as pd
import polars as pl
from pyrsm.utils import ifelse
from pyrsm.example_data import load_data


def get_dfs(pkg=None, name=None, polars=False):
    obj = pd.DataFrame if not polars else pl.DataFrame
    data_dct = {k: v for k, v in globals().items() if isinstance(v, obj)}
    descriptions_dct = {
        k: globals()[f"{k}_description"]
        for k in data_dct
        if f"{k}_description" in globals()
    }
    if len(data_dct) == 0 and name is not None:
        data, description = load_data(pkg=pkg, name=name, polars=polars)
        data_dct = {name: data}
        descriptions_dct = {name: description}
    elif len(data_dct) == 0 and pkg is not None:
        data_dct, descriptions_dct = load_data(pkg=pkg)

    return ifelse(len(data_dct) == 0, None, data_dct), ifelse(
        len(descriptions_dct) == 0, None, descriptions_dct
    )


def message():
    print(
        "Pyrsm and Radiant are open source tools and free to use. If you\nare a student or instructor using pyrsm or Radiant for a class,\nas a favor to the developers, please send an email to\n<radiant@rady.ucsd.edu> with the name of the school and class.\nIf you are using Radiant in your company, as a favor to the\ndeveloper, please share the name of your company and what types\nof activites you are supporting with the tool."
    )


def head_content():
    """
    Return the head content for the shiny app
    """
    return ui.head_content(
        # from https://github.com/rstudio/py-shiny/issues/491#issuecomment-1579138681
        ui.tags.link(
            href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/agate.min.css",
            rel="stylesheet",
        ),
        ui.tags.link(rel="shortcut icon", href="/www/imgs/icon.png"),
        ui.tags.link(href="/www/style.css", rel="stylesheet"),
        ui.tags.script(src="/www/js/returnTextAreaBinding.js"),
        ui.tags.script(src="/www/js/radiantUI.js"),
        ui.tags.script(src="/www/js/screenshot.js"),
        ui.tags.script(
            src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js",
        ),
        ui.tags.script(
            src="//html2canvas.hertzen.com/dist/html2canvas.min.js",
        ),
    )


def init(self, datasets, descriptions=None, state=None, code=True, navbar=None):
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
    self.code = code  # keep code windows open or closed by default
    self.stop_code = ""
    if state is None:
        self.state = {}
    else:
        self.state = state
    if navbar is None:
        self.navbar = ()
    else:
        self.navbar = navbar


def escape_quotes(cmd):
    return cmd.replace('"', '\\"').replace("'", "\\'")


def quote(v, k, ignore=["data"]):
    if isinstance(v, str) and k not in ignore and not (v[0] + v[-1] == "{}"):
        v = escape_quotes(v)
        return f'"{v}"'
    else:
        return v


def copy_icon(cmd, id):
    cmd = escape_quotes(cmd).replace("\n", "\\n")
    return (
        ui.input_action_link(
            id,
            None,
            icon=icon_svg("copy", width="1.5em", height="1.5em"),
            title="Copy to clipboard",
            onclick=f'copyToClipboard("{cmd}");',
        ),
    )


def copy_reset(input, session, id="copy"):
    @reactive.Effect
    @reactive.event(input[id], ignore_none=True)
    async def copy_success():
        ui.update_action_link(
            id,
            icon=icon_svg("check", fill="green", width="1.5em", height="1.5em"),
        )
        # await session.send_custom_message("reset_copy", "")

    @reactive.Effect
    @reactive.event(input.copy_reset, ignore_none=True)
    async def copy_reset():
        req(input[id])
        ui.update_action_link(
            id,
            icon=icon_svg("copy", width="1.5em", height="1.5em"),
        )


def code_formatter(code, self, input, session, id="copy"):
    """
    Format python code using black
    """
    cmd = self.stop_code = black.format_str(code, mode=black.Mode())
    copy_reset(input, session, id=id)
    return ui.TagList(
        ui.HTML(
            f"<details {ifelse(self.code, 'open', '')}><summary>View generated python code</summary>"
        ),
        copy_icon(cmd, id=id),
        ui.markdown(f"""\n```python\n{cmd.rstrip()}\n```"""),
        ui.tags.script("hljs.highlightAll();"),
        ui.HTML("</details>"),
        ui.br(),
    )


def drop_default_args(args, func, ignore=["data"]):
    """
    Take a dictionary of arguments for a function and compare
    to the default arguments for that function
    Return a comma separate string of key-value pairs for
    non-default values where the value is quoted if it is a string
    """
    sig = inspect.signature(func).parameters
    keep = {k: args[k] for k, v in sig.items() if k in args and args[k] != v.default}

    return ", ".join(f"{k}={quote(v, k, ignore)}" for k, v in keep.items())


def make_side_by_side(a, b):
    """Put two inputs side-by-side in the UI"""
    return ui.tags.table(
        ui.tags.td(a, width="50%"), ui.tags.td(b, width="50%"), width="100%"
    )


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


def make_data_elements(self, input, output, session):
    @output(id="show_data_code")
    @render.ui
    def show_data_code():
        return code_formatter(
            get_data(self, input)["code"], self, input, session, id="copy_data"
        )

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

    @reactive.Calc
    def reactive_get_data():
        return get_data(self, input)

    return reactive_get_data


def check_input_value(k, i):
    """
    The '... in input' approach fails with some regulariy (see app_state.py).
    see https://discord.com/channels/1109483223987277844/1127817202804985917/1129530385429176371
    Also tuples need to be converted to lists to be picked up on refresh. Not
    clear why that is needed but it is
    """
    try:
        # print(f"Working on {k}")
        value = i[k]()
        if isinstance(value, tuple):
            return list(value)
        else:
            return value
    except Exception as err:
        # print(f"Extracting value for {k} failed")
        print(err)  # no error message printed
        return None


def dct_update(self, input):
    input_keys = [k for k in input.__dict__["_map"].keys() if k[0] != "."]
    self.state.update({k: check_input_value(k, input) for k in input_keys})


# def update_state():
#     with reactive.isolate():
#         avoid = ["copy_reset"]  # issue because async?
#         input_keys = [
#             k
#             for k in input.__dict__["_map"].keys()
#             if k[0] != "." and k not in avoid
#         ]
#         # print(type(input.copy_reset())) # can't even run this without an error
#         self.state.update({k: input[k]() for k in input_keys})
#         self.state.update(
#             {k: list(v) for k, v in self.state.items() if isinstance(v, tuple)}
#         )


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


def ui_view(self):
    return ui.panel_well(
        ui.input_select(
            "datasets",
            "Datasets:",
            self.dataset_list,
            selected=self.state.get("datasets", None),
        ),
        ui.input_checkbox(
            "show_filter", "Show data filter", value=self.state.get("show_filter", True)
        ),
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
                value=self.state.get("data_filter", ""),
                placeholder="Provide a filter (e.g., price > 5000) and press return",
            ),
            input_return_text_area(
                "data_sort",
                "Data sort:",
                rows=2,
                value=self.state.get("data_sort", ""),
                placeholder="Sort (e.g., ['color', 'price'], ascending=[True, False])) and press return",
            ),
            input_return_text_area(
                "data_slice",
                "Data slice (rows):",
                rows=1,
                value=self.state.get("data_slice", ""),
                placeholder="e.g., 0:50 and press return",
            ),
        ),
    )


def ui_data(self):
    return ui.panel_conditional("input.tabs == 'Data'", ui_view(self))


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


def ui_plot(self, choices, *args):
    return ui.panel_conditional(
        "input.tabs == 'Plot'",
        ui.panel_well(
            ui.input_select(
                id="plots",
                label="Plots",
                selected=self.state.get("plots", None),
                choices=choices,
            ),
            *args,
        ),
    )


def ui_data_main():
    return ui.div(
        ui.output_ui("show_data_code"),
        ui.output_data_frame("show_data"),
        ui.output_ui("show_description"),
    )


def ui_main_basics(height="500px", width="700px"):
    return ui.navset_tab_card(
        ui.nav(
            "Data",
            ui_data_main(),
        ),
        ui.nav(
            "Summary",
            ui.output_ui("show_summary_code"),
            ui.output_text_verbatim("summary"),
        ),
        ui.nav(
            "Plot",
            ui.output_ui("show_plot_code"),
            ui.output_ui("plot_container"),
        ),
        id="tabs",
    )


def ui_main_model():
    return ui.navset_tab_card(
        ui.nav(
            "Data",
            ui_data_main(),
        ),
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
            # ui.output_plot("plot", height="800px", width="700px"),
            ui.output_ui("plot_container"),
        ),
        id="tabs",
    )


def radiant_navbar():
    return (
        ui.nav_control(
            ui.input_action_link("data", "Data", onclick='window.location.href = "/";')
        ),
        ui.nav_menu(
            "Basics",
            ui.nav_control(
                # "Probability",
                ui.input_action_link(
                    "pc",
                    "Probability Calculator",
                    # onclick='window.location.href = "/basics/prob-calc/?SSUID=local-e47b75";',
                    onclick='window.location.href = "/basics/prob-calc/";',
                ),
                # "-----",
                # "Means",
                ui.input_action_link(
                    "sm",
                    "Single mean",
                    onclick='window.location.href = "/basics/single-mean/";',
                ),
                ui.input_action_link(
                    "cm",
                    "Compare means",
                    onclick='window.location.href = "/basics/compare-means/";',
                ),
                # "----",
                # "Proportions",
                ui.input_action_link(
                    "sm",
                    "Single proportion",
                    onclick='window.location.href = "/basics/single-prop/";',
                ),
                # "----",
                # "Tables",
                ui.input_action_link(
                    "gt",
                    "Goodness of fit",
                    onclick='window.location.href = "/basics/goodness/";',
                ),
                ui.input_action_link(
                    "ct",
                    "Cross-tabs",
                    onclick='window.location.href = "/basics/cross-tabs/";',
                ),
            ),
        ),
        ui.nav_menu(
            "Models",
            ui.nav_control(
                ui.input_action_link(
                    "reg",
                    "Linear Regression (OLS)",
                    onclick='window.location.href = "/models/regress/";',
                ),
                ui.input_action_link(
                    "lr",
                    "Logistic Regression (GLM)",
                    onclick='window.location.href = "/models/logistic/";',
                ),
            ),
        ),
    )


def ui_stop():
    return (
        # ui.nav_control(
        #     ui.input_action_link(
        #         "screenshot",
        #         "Screenshot",
        #         icon=icon_svg("camera"),
        #         onclick="generate_screenshot();",
        #     ),
        # ),
        ui.nav_control(
            ui.input_action_link(
                "stop", "Stop", icon=icon_svg("stop"), onclick="window.close();"
            ),
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

    return run_refresh, run_done  # , copy_success, copy_reset


# radiant_screenshot_modal <- function(report_on = "") {
#   add_button <- function() {
#     if (is.empty(report_on)) {
#       ""
#     } else {
#       actionButton(report_on, "Report", icon = icon("edit", verify_fa = FALSE), class = "btn-success")
#     }
#   }
#   showModal(
#     modalDialog(
#       title = "Radiant screenshot",
#       span(shiny::tags$div(id = "screenshot_preview")),
#       span(HTML("</br>To include a screenshot in a report first save it to disk by clicking on the <em>Save</em> button. Then click the <em>Report</em> button to insert a reference to the screenshot into <em>Report > Rmd</em>.")),
#       footer = tagList(
#         tags$table(
#           tags$td(download_button("screenshot_save", "Save", ic = "download")),
#           tags$td(add_button()),
#           tags$td(modalButton("Cancel")),
#           align = "right"
#         )
#       ),
#       size = "l",
#       easyClose = TRUE
#     )
#   )
# }

# observeEvent(input$screenshot_link, {
#   radiant_screenshot_modal()
# })

# render_screenshot <- function() {
#   plt <- sub("data:.+base64,", "", input$img_src)
#   png::readPNG(base64enc::base64decode(what = plt))
# }

# download_handler_screenshot <- function(path, plot, ...) {
#   plot <- try(plot(), silent = TRUE)
#   if (inherits(plot, "try-error") || is.character(plot) || is.null(plot)) {
#     plot <- ggplot() +
#       labs(title = "Plot not available")
#     png(file = path, width = 500, height = 100, res = 96)
#     print(plot)
#     dev.off()
#   } else {
#     ppath <- parse_path(path, pdir = getOption("radiant.launch_dir", find_home()), mess = FALSE)
#     # r_info[["latest_screenshot"]] <- glue("![]({ppath$rpath})")
#     # r_info[["latest_screenshot"]] <- glue("<details>\n<summary>Click to show screenshot</summary>\n<img src='{ppath$rpath}' alt='Radiant screenshot'>\n</details>")
#     r_info[["latest_screenshot"]] <- glue("\n<details>\n<summary>Click to show screenshot with Radiant settings to generate output shown below</summary>\n\n![]({ppath$rpath})\n</details></br>\n")
#     png::writePNG(plot, path, dpi = 144)
#   }
# }

# download_handler(
#   id = "screenshot_save",
#   fun = download_handler_screenshot,
#   fn = function() paste0(r_info[["radiant_tab_name"]], "-screenshot"),
#   type = "png",
#   caption = "Save radiant screenshot",
#   plot = render_screenshot,
#   btn = "button",
#   label = "Save",
#   class = "btn-primary",
#   onclick = "get_img_src();"
# )
