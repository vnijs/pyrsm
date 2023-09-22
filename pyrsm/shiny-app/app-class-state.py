from shiny import App, ui, reactive, render
from starlette.applications import Starlette
from starlette.routing import Mount
import pandas as pd
from starlette.requests import Request as StarletteRequest


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


def ui_view(self):
    return ui.input_text_area(
        "data_filter",
        "Data Filter:",
        rows=2,
        value=self.state.get("data_filter", ""),
        placeholder="Provide a filter (e.g., a > 1) and press return",
    )


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


def get_data_fun(self, input):
    @reactive.Calc
    def reactive_get_data():
        if input is not None and input.data_filter() != "":
            return self.data.query(input.data_filter())
        else:
            return self.data

    return reactive_get_data


class radiant_data:
    def __init__(self, data=None, state=None) -> None:
        if state is None:
            self.state = {}
        else:
            self.state = state
        self.data = data
        self.test = 0

    def shiny_ui(self, request: StarletteRequest):
        return ui.page_navbar(
            ui.nav(
                None,
                ui.row(
                    ui.column(3, ui_view(self)),
                    ui.column(8, ui.output_data_frame("show_data")),
                ),
            ),
            radiant_navbar(),
            title="Radiant for Python",
            inverse=True,
            id="navbar_id",
        )

    def shiny_server(self, input, output, session):
        def update_state():
            with reactive.isolate():
                dct_update(self, input)

        session.on_ended(update_state)

        ## can we make this accessible to the sub_app through super()?
        self.get_data = get_data_fun(self, input)

        @output(id="show_data")
        @render.data_frame
        def show_data():
            self.test = 3
            return render.DataTable(self.get_data())


class radiant_sub_app(radiant_data):
    # class should be able to take in data directly in case
    # it is run in isolation (i.e., without radiant data)
    # however, if not data is passed in and it is used together
    # with radiant_data the data shown below should be coming from
    # radiant_data, ideally from a reactive calc in radiant_data
    # def __init__(self, data=None, state=None) -> None:
    # def __init__(self, data=None, state=None) -> None:
    def __init__(self, rc=None, data=None, state=None) -> None:
        if rc is None:
            if state is None:
                self.state = {}
            else:
                self.state = state
            self.data = data

        else:
            super().__init__(rc.data, rc.state)  # , rc.test, rc.get_data)
            # dct = {attr: getattr(self, attr) for attr in vars(self)}
            # print(dct)
            # self.test = 1
            # print(self.test)
            # self.get_data

            # dct = {attr: getattr(rc, attr) for attr in vars(rc)}
            # super().__init__(**{attr: getattr(rc, attr) for attr in vars(rc)})

    def shiny_ui(self, request: StarletteRequest):
        return ui.page_navbar(
            ui.nav(
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
            inverse=True,
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

        # self.get_data = get_data_fun(self, None)
        print("self.test is", self.test)

        @output(id="show_data")
        @render.data_frame
        def show_data():
            print("self.test is", self.test)
            # return render.DataTable(super().get_data())
            # return render.DataTable(super().data)
            # print(super().__init__())
            # return render.DataTable(self.get_data())
            return render.DataTable(self.data)


data1 = pd.DataFrame().assign(
    a=[1, 2, 3, 4], b=[4, 5, 6, 7], c=["w", "x", "y", "z"], d=[100, -100, 0, -1]
)

data2 = pd.DataFrame().assign(
    a=[1, 2, 3], b=[4, 5, 6], c=["x", "y", "z"], d=[100, -100, 0]
)


rc = radiant_data(data1, state=None)
# rc_sub = radiant_sub_app(data2, state=None)
rc_sub = radiant_sub_app(rc)  # , data=data2, state=None)

routes = [
    Mount("/sub-app", app=App(rc_sub.shiny_ui, rc_sub.shiny_server)),
    Mount("/", app=App(rc.shiny_ui, rc.shiny_server)),
]
app = Starlette(debug=True, routes=routes)
