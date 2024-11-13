from datetime import datetime
from shiny import App, ui, reactive, render
from starlette.requests import Request as StarletteRequest

state = {"__pending_changes__": False}
print("Do we get back to global after refresh?", state)  # we don't


def app_ui(request: StarletteRequest):
    if state["__pending_changes__"]:
        # from https://gist.github.com/jcheng5/88cb05e39c44704ae89e6559130e7f80
        # by Joe Cheng (Posit)
        print("There were pending changes")
        # redirect to the current URL, including query string and hash
        print(
            "Pending if-statement called at: " + datetime.now().strftime("%H:%M:%S.%f")
        )
        return ui.tags.html(
            ui.tags.meta({"http-equiv": "refresh", "content": "0"}),
            ui.tags.script("""window.Shiny.createSocket = function() {};"""),
        )

    print("Do we get back to app_ui after browser refresh?", state)  # we do
    print("Full ui re-render at: " + datetime.now().strftime("%H:%M:%S.%f"))
    return ui.page_fluid(
        ui.navset_tab(
            ui.nav_panel(
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

    # still needs `on_flushed` so we can tell if anything changed
    def update_state():
        print("On-ended function called at: " + datetime.now().strftime("%H:%M:%S.%f"))
        with reactive.isolate():
            dct_update(input)
            state["__pending_changes__"] = False

        print(
            "On-ended function finished at: " + datetime.now().strftime("%H:%M:%S.%f")
        )

    session.on_ended(update_state)

    # session.on_flushed sets __pending_changes__ to True if any changes
    # were made. note that this does NOT update the state dictionary
    # the __pending_changes__ mechanism is needed because otherwise the
    # state update would not be completed before app_ui is re-rendered
    def on_flushed():
        print("on_flushed called")
        state["__pending_changes__"] = True

    session.on_flushed(on_flushed, once=False)

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
