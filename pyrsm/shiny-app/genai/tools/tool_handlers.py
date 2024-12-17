import json
# import pandas as pd
# from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
import base64
from pyrsm.basics import single_mean, compare_means
import pyrsm as rsm


async def handle_single_mean(tool_call, chat):
        try:
            arguments = json.loads(tool_call.function.arguments)
            arguments["data"] = f"{{\"{arguments['data']}\": {arguments['data']}}}"

            cmd = (
                f"single_mean("
                + ", ".join(
                    [
                        f"{k}={v}" if isinstance(v, (int, float)) or k == "data" else f'{k}="{v}"'
                        for k, v in arguments.items()
                        if k not in ["dec"]
                    ]
                )
                + ")"
            )

            # Execute the analysis
            obj = eval(cmd)

            # Extract additional arguments for summary method if provided
            dec = arguments.pop("dec", 3)

            # Get summary
            buffer_summary = io.StringIO()
            with redirect_stdout(buffer_summary):
                obj.summary(dec=dec)

            # Generate and send plot
            plt.figure()  # Create new figure
            obj.plot()
            buffer_plot = io.BytesIO()
            plt.savefig(buffer_plot, format="png")
            plt.close("all")  # Close all figures
            buffer_plot.seek(0)
            encoded_image = base64.b64encode(buffer_plot.read()).decode("utf-8")

            cmd = f"sm = {cmd}"
            cmd += f"\nsm.summary(dec={dec})"
            cmd += "\nsm.plot(plots=\"hist\")"

            results = f"Generated code:\n```python\n{cmd}\n```\n"
            results += f"\nSummary output:\n```bash\n{buffer_summary.getvalue()}```\n"
            results += f"![plot](data:image/png;base64,{encoded_image})"
            await chat.append_message({"role": "assistant", "content": results})

        except Exception as e:
            error_msg = f"Error executing analysis: {str(e)}"
            await chat.append_message({"role": "assistant", "content": error_msg})
        finally:
            plt.close("all")  # Ensure all plots are closed


async def handle_compare_means(tool_call, chat):
    try:
        arguments = json.loads(tool_call.function.arguments)
        arguments["data"] = f"{{\"{arguments['data']}\": {arguments['data']}}}"

        cmd = (
            f"compare_means("
            + ", ".join(
                [
                    f"{k}={v}" if isinstance(v, (int, float)) or k == "data" else f'{k}="{v}"'
                    for k, v in arguments.items()
                    if k not in ["dec", "extra", "plots", "nobs"]
                ]
            )
            + ")"
        )

        # Execute the analysis
        obj = eval(cmd)

        # Extract additional arguments for summary method if provided
        dec = arguments.pop("dec", 3)
        extra = arguments.pop("extra", False)

        # Get summary
        buffer_summary = io.StringIO()
        with redirect_stdout(buffer_summary):
            obj.summary(extra=extra, dec=dec)

        # Extract plot type if provided in the arguments
        plots = arguments.pop("plots", "scatter")
        nobs = arguments.pop("nobs", None)

        # Generate and send plot
        plt.figure()  # Create new figure
        obj.plot(plots=plots, nobs=nobs)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close("all")  # Close all figures
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")

        cmd = f"cm = {cmd}"
        cmd += f"\ncm.summary(extra={extra}, dec={dec})"
        if plots in ["scatter", "box"]:
            cmd += f"\ncm.plot(plots=\"{plots}\", nobs={nobs})"
        else:
            cmd += f"\ncm.plot(plots=\"{plots}\")"

        results = f"Generated code:\n```python\n{cmd}\n```\n"
        results += f"\nSummary output:\n```bash\n{buffer_summary.getvalue()}```\n"
        results += f"![plot](data:image/png;base64,{encoded_image})"

        await chat.append_message({"role": "assistant", "content": results})

    except Exception as e:
        error_msg = f"Error executing analysis: {str(e)}"
        await chat.append_message({"role": "assistant", "content": error_msg})
    finally:
        plt.close("all")  # Ensure all plots are closed


async def handle_linear_regression(tool_call, chat):
    try:
        arguments = json.loads(tool_call.function.arguments)
        arguments["data"] = f"{{\"{arguments['data']}\": {arguments['data']}}}"

        cmd = (
            f"regress("
            + ", ".join(
                [
                    f"{k}={v}" if isinstance(v, (int, float)) or k == "data" else f'{k}="{v}"'
                    for k, v in arguments.items()
                    if k not in ["dec", "extra", "plots", "nobs"]
                ]
            )
            + ")"
        )

        # Execute the analysis
        obj = eval(cmd)

        # Extract additional arguments for summary method if provided
        dec = arguments.pop("dec", 3)

        # Get summary
        buffer_summary = io.StringIO()
        with redirect_stdout(buffer_summary):
            obj.summary(dec=dec)

        # Extract plot type if provided in the arguments
        plots = arguments.pop("plots", "scatter")
        nobs = arguments.pop("nobs", None)

        # Generate and send plot
        plt.figure()  # Create new figure
        obj.plot(plots=plots, nobs=nobs)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close("all")  # Close all figures
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")

        cmd = f"reg = {cmd}"
        cmd += f"\nreg.summary(dec={dec})"
        if plots in ["scatter", "box"]:
            cmd += f"\nreg.plot(plots=\"{plots}\", nobs={nobs})"
        else:
            cmd += f"\nreg.plot(plots=\"{plots}\")"

        results = f"Generated code:\n```python\n{cmd}\n```\n"
        results += f"\nSummary output:\n```bash\n{buffer_summary.getvalue()}```\n"
        results += f"![plot](data:image/png;base64,{encoded_image})"

        await chat.append_message({"role": "assistant", "content": results})

    except Exception as e:
        error_msg = f"Error executing analysis: {str(e)}"
        await chat.append_message({"role": "assistant", "content": error_msg})
    finally:
        plt.close("all")  # Ensure all plots are closed
