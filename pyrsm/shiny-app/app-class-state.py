import pandas as pd
from shiny import App, reactive, render, ui
from starlette.applications import Starlette
from starlette.requests import Request as StarletteRequest
from starlette.routing import Mount


def check_input_value(k, i):
    """
    The '... in input' approach causes issues (see app_state.py).
    see https://discord.com/channels/1109483223987277844/1127817202804985917/1129530385429176371
    Also tuples need to be converted to lists to be picked up on refresh by select inputs. Not
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


def radiant_navbar():
    return (
        ui.nav_control(
            ui.input_action_link("data", "Data", onclick='window.location.href = "/";')
        ),
        ui.nav_control(
            ui.input_action_link(
                "sa",
                "Sub App",
                onclick='window.location.href = "/sub-app/";',
            ),
        ),
    )


class radiant_data:
    def __init__(self, data, state=None) -> None:
        if state is None:
            self.state = {}
        else:
            self.state = state
        self.data = data

    def shiny_ui(self, request: StarletteRequest):
        return ui.page_navbar(
            ui.nav_panel(
                None,
                ui.row(
                    ui.column(
                        3,
                        ui.input_text_area(
                            "data_filter",
                            "Data Filter:",
                            rows=2,
                            value=self.state.get("data_filter", ""),
                            placeholder="Provide a filter (e.g., a > 1) and press return",
                        ),
                    ),
                    ui.column(8, ui.output_data_frame("show_data")),
                ),
            ),
            radiant_navbar(),
            title="Radiant for Python",
            inverse=False,
            id="navbar_id",
        )

    def shiny_server(self, input, output, session):
        def update_state():
            with reactive.isolate():
                dct_update(self, input)

        session.on_ended(update_state)

        @reactive.Calc
        def get_data():
            if input.data_filter() != "":
                try:
                    filtered_data = self.data.query(input.data_filter())
                    return filtered_data
                except Exception:
                    return self.data
            else:
                return self.data

        # self.get_data will be accessible to sub-apps as an
        # attribute of the radiant_data class instance
        # but *only* after the user has visited the Data tab
        # and self.get_data has been intialized
        self.get_data = get_data

        @output(id="show_data")
        @render.data_frame
        def show_data():
            return render.DataTable(self.get_data())


class radiant_sub_app:
    def __init__(self, data=None, state=None) -> None:
        if state is None:
            self.state = {}
        else:
            self.state = state

        if data is not None:
            self.data = data

    def shiny_ui(self, request: StarletteRequest):
        return ui.page_navbar(
            ui.nav_panel(
                None,
                ui.row(
                    ui.column(
                        3,
                        ui.input_slider("n", "N", 0, 100, self.state.get("n", 20)),
                        ui.input_select(
                            "var1",
                            label="Variables 1:",
                            selected=self.state.get("var1", None),
                            choices=["a", "b", "c"],
                            size=3,
                            multiple=True,
                        ),
                        ui.output_ui("ui_var2"),
                    ),
                    ui.column(8, ui.output_data_frame("show_data")),
                ),
            ),
            radiant_navbar(),
            title="Radiant for Python",
            inverse=False,
            id="navbar_id",
        )

    def shiny_server(self, input, output, session):
        def update_state():
            with reactive.isolate():
                dct_update(self, input)

        session.on_ended(update_state)

        @output(id="ui_var2")
        @render.ui
        def ui_var2():
            return ui.input_select(
                "var2",
                label="Variables 2 (dynamic):",
                selected=self.state.get("var2", None),
                choices=["a", "b", "c"],
                size=3,
                multiple=True,
            )

        @output(id="show_data")
        @render.data_frame
        def show_data():
            if hasattr(rc, "get_data"):
                return render.DataTable(rc.get_data())
            else:
                return render.DataTable(rc.data)


data = pd.DataFrame().assign(
    a=[1, 2, 3, 4], b=[4, 5, 6, 7], c=["w", "x", "y", "z"], d=[100, -100, 0, -1]
)

rc = radiant_data(data, state=None)
rc_sub = radiant_sub_app(data=None, state=None)

routes = [
    Mount("/sub-app", app=App(rc_sub.shiny_ui, rc_sub.shiny_server)),
    Mount("/", app=App(rc.shiny_ui, rc.shiny_server)),
]
app = Starlette(debug=True, routes=routes)
