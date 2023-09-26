from datetime import datetime
from shiny import App, ui, reactive, render
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

        print(
            "On-ended function finished at: " + datetime.now().strftime("%H:%M:%S.%f")
        )

    # would be perfect *if* if would complete before app_ui is re-rendered
    session.on_ended(update_state)

    # session.on_flushed does keep the state updated but might
    # be inefficient if there are many large inputs (e.g., data) that are
    # updated when any input changes
    # session.on_flushed(update_state, once=False)

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


app = App(app_ui, server)
