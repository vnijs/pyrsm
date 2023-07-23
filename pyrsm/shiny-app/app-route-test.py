from starlette.applications import Starlette
from starlette.routing import Mount
from shiny import App, ui

app_shiny = App(ui.page_fluid("hello from shiny!"), None)
app_home = App(ui.page_fluid("hello from home!"), None)


def radiant_navbar():
    return ui.page_navbar(
        ui.nav_control(
            ui.input_action_link("home", "Home", onclick='window.location.href = "/";')
        ),
        ui.nav_menu(
            "Basics",
            ui.nav_control(
                ui.input_action_link(
                    "sm",
                    "Single mean",
                    onclick='window.location.href = "/basics/single-mean/";',
                )
            ),
        ),
        ui.nav_menu(
            "Models",
            ui.nav_control(
                ui.input_action_link(
                    "reg",
                    "Regress",
                    onclick='window.location.href = "/models/regress/";',
                )
            ),
        ),
        title="Radiant for Python",
        inverse=True,
        id="navbar_id",
    )


routes = [
    Mount("/home", app=app_home),
    Mount("/shiny", app=app_shiny),
    Mount("/", app=app_home),  # must be last!
]

app = Starlette(routes=routes, debug=True)
