from shiny import App, Inputs, Outputs, Session, ui, reactive, render
import shinyswatch

# based on https://github.com/posit-dev/py-shinyswatch/issues/11#issuecomment-1647868499
from starlette.requests import Request as StarletteRequest

if "theme" not in globals():
    print("Define theme dictionary. Was not in globals()")
    theme = {"theme": "superhero"}


def app_ui(request: StarletteRequest):
    print("ui function was called")
    return ui.page_fluid(
        shinyswatch.get_theme(theme.get("theme", "superhero")),
        ui.tags.script(
            """
            Shiny.addCustomMessageHandler('refresh', function(message) {
                window.location.reload();
            });
            """
        ),
        ui.input_select(
            id="select_theme",
            label="Select a theme:",
            selected=theme.get("theme", "superhero"),
            choices=["superhero", "darkly", "sketchy"],
        ),
        # ui.output_ui("ui_select_theme"),
    )


def server(input: Inputs, output: Outputs, session: Session):
    print("server function was called")
    print(theme.get("theme", "superhero"))

    @reactive.Effect
    @reactive.event(input.select_theme, ignore_none=True)
    async def set_theme():
        if input.select_theme() != theme.get("theme", "superhero"):
            theme["theme"] = input.select_theme()
            await session.send_custom_message("refresh", "")
        # ui.update_select("select_theme", selected=theme.get("theme", "superhero"))

    # @output(id="ui_select_theme")
    # @render.ui
    # def ui_select_theme():
    #     return (
    #         ui.input_select(
    #             id="select_theme",
    #             label="Select a theme:",
    #             selected=theme.get("theme", "superhero"),
    #             choices=["superhero", "darkly", "sketchy"],
    #         ),
    #     )


app = App(app_ui, server, debug=False)
