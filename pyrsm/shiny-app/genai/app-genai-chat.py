import os
import json
import pandas as pd
from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
import base64
import traceback
import inspect

from app_utils import load_dotenv
from openai import AsyncOpenAI
from pyrsm.basics import single_mean, compare_means
import pyrsm as rsm

load_dotenv()
llm = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load default data
base_dir = "pyrsm/data/"
demand_uk = pd.read_parquet(f"{base_dir}/basics/demand_uk.parquet")
with open(f"{base_dir}/basics/demand_uk_description.md", "r") as f:
    demand_uk_description = f.read()

salary = pd.read_parquet(f"{base_dir}/basics/salary.parquet")
with open(f"{base_dir}/basics/salary_description.md", "r") as f:
    salary_description = f.read()

# JSON representation of the single_mean class
single_mean_tool = {
    "name": "single_mean",
    "description": "Perform single-mean hypothesis testing.",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": "The name of the dataframe available in the environment to be used for hypothesis testing.",
            },
            "var": {"type": "string", "description": "The variable/column name to test."},
            "alt_hyp": {
                "type": "string",
                "enum": ["two-sided", "greater", "less"],
                "description": "The alternative hypothesis.",
            },
            "conf": {"type": "number", "description": "The confidence level for the test.", "default": 0.95},
            "comp_value": {"type": "number", "description": "The comparison value for the test.", "default": 0},
        },
        "required": ["data", "var"],
    },
    "attributes": {
        "data": {
            "type": "pd.DataFrame | pl.DataFrame",
            "description": "The input data for the hypothesis test as a Pandas or Polars DataFrame.",
        },
        "var": {"type": "string", "description": "The variable/column name to test."},
        "alt_hyp": {
            "type": "string",
            "description": "The alternative hypothesis ('two-sided', 'greater', 'less').",
        },
        "conf": {"type": "number", "description": "The confidence level for the test."},
        "comp_value": {"type": "number", "description": "The comparison value for the test."},
        "t_val": {"type": "number", "description": "The t-statistic value."},
        "p_val": {"type": "number", "description": "The p-value of the test."},
        "ci": {"type": "tuple", "description": "The confidence interval of the test."},
        "mean": {"type": "number", "description": "The mean of the variable."},
        "n": {"type": "integer", "description": "The number of observations."},
        "n_missing": {"type": "integer", "description": "The number of missing observations."},
        "sd": {"type": "number", "description": "The standard deviation of the variable."},
        "se": {"type": "number", "description": "The standard error of the variable."},
        "me": {"type": "number", "description": "The margin of error."},
        "diff": {"type": "number", "description": "The difference between the mean and the comparison value."},
        "df": {"type": "integer", "description": "The degrees of freedom."},
    },
    "methods": {
        "__init__": {
            "description": "Initializes the single_mean class with the provided data and parameters.",
            "parameters": {
                "data": {
                    "type": "pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame]",
                    "description": "The input data for the hypothesis test as a Pandas or Polars DataFrame or a dictionary of dataframes.",
                },
                "var": {"type": "string", "description": "The variable/column name to test."},
                "alt_hyp": {
                    "type": "string",
                    "default": "two-sided",
                    "description": "The alternative hypothesis ('two-sided', 'greater', 'less').",
                },
                "conf": {
                    "type": "number",
                    "default": 0.95,
                    "description": "The confidence level for the test.",
                },
                "comp_value": {
                    "type": "number",
                    "default": 0,
                    "description": "The comparison value for the test.",
                },
            },
        },
        "summary": {
            "description": "Prints a summary of the hypothesis test.",
            "parameters": {
                "dec": {
                    "type": "integer",
                    "default": 3,
                    "description": "The number of decimal places to display in the summary.",
                }
            },
        },
        "plot": {
            "description": "Plots the results of the hypothesis test. The black lines in the histogram show the sample mean (solid line) and the confidence interval around the sample mean (dashed lines). The red line shows the comparison value (i.e., the value under the null-hypothesis). If the red line does not fall within the confidence interval we can reject the null-hypothesis in favor of the alternative at the specified confidence level (e.g., 0.95).",
            "parameters": {
                "plots": {
                    "type": "string",
                    "default": "hist",
                    "enum": ["hist", "sim"],
                    "description": "The type of plot to display ('hist' or 'sim').",
                }
            },
        },
    },
}

# JSON representation of the compare_means class
compare_means_tool = {
    "name": "compare_means",
    "description": "A class to perform comparison of means hypothesis testing.",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": "The name of the dataframe available in the environment to be used for hypothesis testing.",
            },
            "var1": {
                "type": "string",
                "description": "The first variable/column name to test. This can be numeric or categorical.",
            },
            "var2": {"type": "string", "description": "The second variable/column name to test. This must be numeric."},
            "comb": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of string tuples representing combinations of variables for comparison. Defaults to all possible combinations.",
                "default": [],
            },
            "alt_hyp": {
                "type": "string",
                "enum": ["two-sided", "greater", "less"],
                "description": "The alternative hypothesis.",
                "default": "two-sided",
            },
            "conf": {"type": "number", "description": "The confidence level for the test.", "default": 0.95},
            "sample_type": {
                "type": "string",
                "enum": ["independent", "paired"],
                "description": "Type of samples ('independent' or 'paired').",
                "default": "independent",
            },
            "adjust": {
                "type": "string",
                "enum": [None, "bonferroni"],
                "description": "Adjustment for multiple testing (e.g., None or 'bonferroni').",
                "default": None,
            },
            "test_type": {
                "type": "string",
                "enum": ["t-test", "wilcox"],
                "description": "The type of test to use ('t-test' or 'wilcox').",
                "default": "t-test",
            },
        },
        "required": ["data", "var1", "var2"],
    },
    "attributes": {
        "data": {
            "type": "pd.DataFrame | pl.DataFrame",
            "description": "The input data for the hypothesis test as a Pandas or Polars DataFrame.",
        },
        "var1": {"type": "string", "description": "The first variable/column name to test."},
        "var2": {"type": "string", "description": "The second variable/column name to test."},
        "comb": {"type": "array", "description": "List of string tuples representing comparisons to make."},
        "alt_hyp": {"type": "string", "description": "The alternative hypothesis ('two-sided', 'greater', 'less')."},
        "conf": {"type": "number", "description": "The confidence level for the test."},
        "sample_type": {"type": "string", "description": "The type of samples ('independent' or 'paired')."},
        "adjust": {"type": "string", "description": "Adjustment for multiple testing."},
        "test_type": {"type": "string", "description": "The type of test ('t-test' or 'wilcox')."},
        "t_val": {"type": "number", "description": "The t-statistic value."},
        "p_val": {"type": "number", "description": "The p-value of the test."},
        "ci": {"type": "tuple", "description": "The confidence interval of the test."},
        "mean1": {"type": "number", "description": "The mean of the first variable."},
        "mean2": {"type": "number", "description": "The mean of the second variable."},
        "n1": {"type": "integer", "description": "The number of observations for the first variable."},
        "n2": {"type": "integer", "description": "The number of observations for the second variable."},
        "sd1": {"type": "number", "description": "The standard deviation of the first variable."},
        "sd2": {"type": "number", "description": "The standard deviation of the second variable."},
        "se": {"type": "number", "description": "The standard error of the difference between the means."},
        "me": {"type": "number", "description": "The margin of error."},
        "diff": {"type": "number", "description": "The difference between the means of the two variables."},
        "df": {"type": "integer", "description": "The degrees of freedom."},
    },
    "methods": {
        "__init__": {
            "description": "Initializes the compare_means class with the provided data and parameters.",
            "parameters": {
                "data": {
                    "type": "pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame]",
                    "description": "The input data for the hypothesis test as a Pandas or Polars DataFrame or a dictionary of dataframes.",
                },
                "var1": {"type": "string", "description": "The first variable/column name to include in the test."},
                "var2": {"type": "string", "description": "The second variable/column name to include in the test."},
                "comb": {
                    "type": "array",
                    "default": [],
                    "description": "List of string tuples representing comparisons to make. Defaults to all possible combinations.",
                },
                "alt_hyp": {
                    "type": "string",
                    "default": "two-sided",
                    "description": "The alternative hypothesis ('two-sided', 'greater', 'less').",
                },
                "conf": {"type": "number", "default": 0.95, "description": "The confidence level for the test."},
                "sample_type": {
                    "type": "string",
                    "default": "independent",
                    "description": "The type of samples ('independent' or 'paired').",
                },
                "adjust": {
                    "type": "string",
                    "default": None,
                    "description": "Adjustment for multiple testing (e.g., None or 'bonferroni').",
                },
                "test_type": {
                    "type": "string",
                    "default": "t-test",
                    "description": "The type of test ('t-test' or 'wilcox').",
                },
            },
        },
        "summary": {
            "description": "Prints a summary of the hypothesis test.",
            "parameters": {
                "dec": {
                    "type": "integer",
                    "default": 3,
                    "description": "The number of decimal places to display in the summary.",
                },
                "extra": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to include extra details in the summary output.",
                },
            },
        },
        "plot": {
            "description": "Plots the results of the hypothesis test.",
            "parameters": {
                "plots": {
                    "type": "string",
                    "enum": ["scatter", "box", "density", "bar"],
                    "description": "The type of plot to create ('scatter', 'box', 'density', 'bar').",
                    "default": "scatter",
                },
                "nobs": {
                    "type": "integer",
                    "description": "The number of observations to plot. Defaults to all available datapoints.",
                    "default": None,
                },
            },
        },
    },
}


tools = [
    {
        "type": "function",
        "function": single_mean_tool,
    },
    {
        "type": "function",
        "function": compare_means_tool,
    }
]

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Data Management"),
        ui.input_file("data_file", "Upload Data (CSV/Parquet)", accept=[".csv", ".parquet"]),
        ui.input_file("desc_file", "Upload Description (MD/TXT)", accept=[".md", ".txt"]),
        ui.input_select("dataset_select", "Select Dataset", choices=["demand_uk", "salary"]),
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

    async def handle_single_mean(tool_call):
        try:
            arguments = json.loads(tool_call.function.arguments)
            arguments["data"] = f"{{\"{arguments['data']}\": {arguments['data']}}}"

            cmd = (
                f"single_mean("
                + ", ".join(
                    [
                        f"{k}={v}" if isinstance(v, (int, float)) or k == "data" else f'{k}="{v}"'
                        for k, v in arguments.items()
                    ]
                )
                + ")"
            )

            await chat.append_message({"role": "assistant", "content": f"Generated code:\n```python\n{cmd}"})

            # Execute the analysis
            obj = eval(cmd)

            # Get summary
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                obj.summary()

            # Send summary
            await chat.append_message(
                {"role": "assistant", "content": f"Summary output:\n```bash\n{buffer.getvalue()}```\n"}
            )

            # Generate and send plot
            plt.figure()  # Create new figure
            obj.plot()
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close("all")  # Close all figures
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
            await chat.append_message(
                {"role": "assistant", "content": f"![plot](data:image/png;base64,{encoded_image})"}
            )

        except Exception as e:
            error_msg = f"Error executing analysis: {str(e)}"
            await chat.append_message({"role": "assistant", "content": error_msg})
        finally:
            plt.close("all")  # Ensure all plots are closed

    async def handle_compare_means(tool_call):
        try:
            arguments = json.loads(tool_call.function.arguments)
            arguments["data"] = f"{{\"{arguments['data']}\": {arguments['data']}}}"

            cmd = (
                f"compare_means("
                + ", ".join(
                    [
                        f"{k}={v}" if isinstance(v, (int, float)) or k == "data" else f'{k}="{v}"'
                        for k, v in arguments.items()
                    ]
                )
                + ")"
            )

            await chat.append_message({"role": "assistant", "content": f"Generated code:\n```python\n{cmd}"})

            # Execute the analysis
            obj = eval(cmd)

            # Get summary
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                obj.summary()

            # Send summary
            await chat.append_message(
                {"role": "assistant", "content": f"Summary output:\n```bash\n{buffer.getvalue()}```\n"}
            )

            # Generate and send plot
            plt.figure()  # Create new figure
            obj.plot()
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close("all")  # Close all figures
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
            await chat.append_message(
                {"role": "assistant", "content": f"![plot](data:image/png;base64,{encoded_image})"}
            )

        except Exception as e:
            error_msg = f"Error executing analysis: {str(e)}"
            await chat.append_message({"role": "assistant", "content": error_msg})
        finally:
            plt.close("all")  # Ensure all plots are closed


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
                    print(tool_call.function.name)
                    if tool_call.function.name == "single_mean":
                        await handle_single_mean(tool_call)
                    elif tool_call.function.name == "compare_means":
                        await handle_compare_means(tool_call)

                    # Get interpretation after tool execution
                    await chat.append_message(
                        {
                            "role": "system",
                            "content": f"Provide a detailed interpretation of the results from calling the {tool_call} function. If a plot was created, please provide an interpretation of the plot as well.",
                        }
                    )
                    interpret_response = await llm.chat.completions.create(
                        model="gpt-4o", messages=chat.messages(format="openai")
                    )
                    # Append the interpretation
                    await chat.append_message(
                        {"role": "assistant", "content": interpret_response.choices[0].message.content}
                    )
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
