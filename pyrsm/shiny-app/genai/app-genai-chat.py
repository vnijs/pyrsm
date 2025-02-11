import os
import pandas as pd
from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import traceback

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "tools")))
import tools.tool_function_definitions as tfd
import tools.tool_handlers as th

from app_utils import load_dotenv
from openai import AsyncOpenAI
from pyrsm.basics import single_mean, compare_means
from pyrsm.model import regress
import pyrsm as rsm

tools = tfd.tools

load_dotenv()
llm = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load default data
base_dir = "pyrsm/data"
demand_uk = pd.read_parquet(f"{base_dir}/basics/demand_uk.parquet")
with open(f"{base_dir}/basics/demand_uk_description.md", "r") as f:
    demand_uk_description = f.read()

salary = pd.read_parquet(f"{base_dir}/basics/salary.parquet")
with open(f"{base_dir}/basics/salary_description.md", "r") as f:
    salary_description = f.read()

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Data Management"),
        ui.input_file("data_file", "Upload Data (CSV/Parquet)", accept=[".csv", ".parquet"]),
        ui.input_file("desc_file", "Upload Description (MD/TXT)", accept=[".md", ".txt"]),
        ui.input_select("dataset_select", "Select Dataset", choices=["salary", "demand_uk"]),
        ui.output_ui("current_description"),
        width=400,
    ),
    ui.navset_tab(ui.nav_panel("Chat", ui.chat_ui("chat")), ui.nav_panel("Data", ui.output_ui("data"))),
    fillable_mobile=True,
)


def server(input):
    # Reactive values to store datasets and descriptions
    datasets = reactive.value({"demand_uk": demand_uk, "salary": salary})
    descriptions = reactive.value({"demand_uk": demand_uk_description, "salary": salary_description})
    available_datasets = reactive.value(["salary", "demand_uk"])

    @reactive.Effect
    @reactive.event(input.data_file)
    def _():
        file_info = input.data_file()
        if not file_info:
            return

        name = os.path.splitext(file_info["name"])[0]
        if file_info["name"].endswith(".csv"):
            df = pd.read_csv(file_info["datapath"])
        else:  # parquet
            df = pd.read_parquet(file_info["datapath"])

        current_datasets = datasets.get()
        current_datasets[name] = df
        datasets.set(current_datasets)

        current_available = available_datasets.get()
        if name not in current_available:
            current_available.append(name)
            available_datasets.set(current_available)

    @reactive.Effect
    @reactive.event(input.desc_file)
    def _():
        file_info = input.desc_file()
        if not file_info:
            return

        name = os.path.splitext(file_info["name"])[0]
        with open(file_info["datapath"], "r") as f:
            desc = f.read()

        current_descriptions = descriptions.get()
        current_descriptions[name] = desc
        descriptions.set(current_descriptions)

    @render.ui
    def current_description():
        selected = input.dataset_select()
        desc_dict = descriptions.get()
        return ui.markdown(desc_dict.get(selected, "No description available"))

    @render.ui
    @reactive.event(input.dataset_select)
    def data():
        selected = input.dataset_select()
        data_dict = datasets.get()
        if selected in data_dict:
            return ui.output_data_frame("selected_data")
        return ui.p("No data selected")

    @render.data_frame
    def selected_data():
        selected = input.dataset_select()
        data_dict = datasets.get()
        return data_dict.get(selected)

    chat = ui.Chat(
        id="chat",
        messages=[
            {"content": "Hello! How can I help you today?", "role": "assistant"},
        ],
    )

    @chat.on_user_submit
    async def _():
        try:
            # Get current dataset and description
            selected = input.dataset_select()
            desc_dict = descriptions.get()
            current_desc = desc_dict.get(selected, "No description available")

            await chat.append_message({"role": "system", "content": f"Available data: {selected} - {current_desc}"})
            messages = chat.messages(format="openai")

            response = await llm.chat.completions.create(
                model="gpt-4o",  # Fixed model name
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # Handle tool calls if present
            tool_calls = message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.function.name == "single_mean":
                        await th.handle_single_mean(tool_call, chat)
                    elif tool_call.function.name == "compare_means":
                        await th.handle_compare_means(tool_call, chat)
                    elif tool_call.function.name == "regress":
                        await th.handle_linear_regression(tool_call, chat)

                    # Get interpretation after tool execution
                    # await chat.append_message(
                    #     {
                    #         "role": "system",
                    #         "content": f"Provide a detailed interpretation of the results from calling the {tool_call} function. If a plot was created, please provide an interpretation of the plot as well.",
                    #     }
                    # )
                    # interpret_response = await llm.chat.completions.create(
                    #     model="gpt-4o", messages=chat.messages(format="openai")
                    # )
                    # # Append the interpretation
                    # await chat.append_message(
                    #     {"role": "assistant", "content": interpret_response.choices[0].message.content}
                    # )
            else:
                # If no tool calls, just append the response content
                await chat.append_message({"role": "assistant", "content": message.content})

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            await chat.append_message({"role": "assistant", "content": error_msg})
        finally:
            plt.close("all")  # Ensure all plots are closed

    # return [ui.update_select("dataset_select", choices=available_datasets)]


app = App(app_ui, server)
