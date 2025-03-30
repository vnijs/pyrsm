import numpy as np
import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

# Create a date range
dates = pd.date_range("20230101", periods=6)

# Create a DataFrame
data = pd.DataFrame(
    {
        "A": np.random.rand(6),  # numeric column
        "B": pd.Categorical(
            ["test", "train", "test", "train", "test", "train"]
        ),  # categorical column
        "C": dates,  # date column
    }
)

app_ui = ui.page_fluid(
    ui.input_select(
        "selection_mode",
        "Selection mode",
        {"none": "(None)", "single": "Single", "multiple": "Multiple"},
        selected="multiple",
    ),
    ui.input_switch("gridstyle", "Grid", True),
    ui.input_switch("fullwidth", "Take full width", True),
    ui.input_switch("fixedheight", "Fixed height", True),
    ui.output_data_frame("grid"),
    ui.panel_fixed(
        ui.output_text_verbatim("detail"),
        right="10px",
        bottom="10px",
    ),
    class_="p-3",
)


def server(input: Inputs, output: Outputs, session: Session):
    df: reactive.Value[pd.DataFrame] = reactive.Value(data)

    @output
    @render.data_frame
    def grid():
        height = 350 if input.fixedheight() else None
        width = "100%" if input.fullwidth() else "fit-content"
        if input.gridstyle():
            return render.DataGrid(
                df(),
                row_selection_mode=input.selection_mode(),
                height=height,
                width=width,
            )
        else:
            return render.DataTable(
                df(),
                row_selection_mode=input.selection_mode(),
                height=height,
                width=width,
            )

    @reactive.Effect
    @reactive.event(input.grid_cell_edit)
    def handle_edit():
        edit = input.grid_cell_edit()
        df_copy = df().copy()
        df_copy.iat[edit["row"], edit["col"]] = edit["new_value"]
        df.set(df_copy)

    @output
    @render.text
    def detail():
        if (
            input.grid_selected_rows() is not None
            and len(input.grid_selected_rows()) > 0
        ):
            # "split", "records", "index", "columns", "values", "table"
            return df().iloc[list(input.grid_selected_rows())]


app = App(app_ui, server)
