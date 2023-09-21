from datetime import datetime
from shiny import App, ui, reactive, render

# based on https://github.com/posit-dev/py-shinyswatch/issues/11#issuecomment-1647868499
from starlette.requests import Request as StarletteRequest

state = {}
print("Do we get back to global after refresh?", state)  # we don't


def app_ui(request: StarletteRequest):
    print("Do we get back to app_ui after browser refresh?", state)  # we do
    return ui.page_fluid(
        ui.navset_tab(
            ui.nav(
                "Main",
                ui.input_slider("n", "N", 0, 100, state.get("n", 20)),
                ui.input_text_area("f", "Filter", value=state.get("f", "")),
                ui.input_select(
                    "var1",
                    label="Variables 1:",
                    selected=state.get("var1", None),
                    choices=["a", "b", "c"],
                    multiple=True,
                ),
                ui.output_ui("ui_var2"),
                ui.div(
                    "This page was rendered at ", datetime.now().strftime("%H:%M:%S.%f")
                ),
            ),
            ui.nav(
                "Hidden",
                ui.input_slider("count", "Count", 0, 100, state.get("count", 10)),
                ui.output_ui("ui_var3"),
            ),
        ),
    )


def server(input, output, session):
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

    def dct_update(input):
        input_keys = [k for k in input.__dict__["_map"].keys() if k[0] != "."]
        state.update({k: check_input_value(k, input) for k in input_keys})

    def update_state():
        print("On-ended function called at: " + datetime.now().strftime("%H:%M:%S.%f"))
        with reactive.isolate():
            dct_update(input)

        # this seems to work for all available inputs, but restores the state from 2 steps back
        # issue posted: https://github.com/posit-dev/py-shiny/issues/323
        # input_keys = input.__dict__["_map"].keys()
        # state.update({k: input[k]() for k in input_keys if k[0] != "."})
        # state.update({k: list(v) for k, v in state.items() if isinstance(v, tuple)})

        print(
            "On-ended function finished at: " + datetime.now().strftime("%H:%M:%S.%f")
        )

    session.on_ended(update_state)

    @output(id="ui_var2")
    @render.ui
    def ui_var2():
        return ui.input_select(
            "var2",
            label="Variables 2 (dynamic):",
            selected=state.get("var2", None),
            choices=["a", "b", "c"],
            multiple=True,
        )

    @output(id="ui_var3")
    @render.ui
    def ui_var3():
        return ui.input_select(
            "var3",
            label="Variables 3 (dynamic):",
            selected=state.get("var3", None),
            choices=["a", "b", "c"],
            multiple=True,
        )


app = App(app_ui, server)
