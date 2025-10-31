"""
MCP Bridge Magic - Connect Jupyter notebooks to LLM tool calling

Usage:
    %load_ext mcp_bridge_magic

    %%mcp
    Compare salary between academic ranks
"""

from IPython.core.magic import register_cell_magic, register_line_magic
from IPython import get_ipython
from IPython.display import display, Markdown, HTML
import ipywidgets as widgets
import pandas as pd
import os
import sys
import asyncio
from pathlib import Path
import re

# Fix for Jupyter: allow nested event loops
import nest_asyncio
nest_asyncio.apply()

# Setup paths
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

import google.generativeai as genai
from server_regression import DATA_STORE, MODEL_STORE, call_tool

# Active dataset tracking (like R/Radiant)
ACTIVE_DATASET = None

# Store last prompt for re-execution (e.g., after dataset selection)
LAST_PROMPT = None


def _load_gemini_api_key():
    """Load Gemini API key from ~/.env"""
    env_file = Path.home() / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("No GEMINI_API_KEY or GOOGLE_API_KEY found in ~/.env")

    genai.configure(api_key=api_key)


def _get_kernel_context():
    """Extract context from IPython kernel namespace"""
    ipython = get_ipython()
    if ipython is None:
        return {}

    user_ns = ipython.user_ns

    # Find DataFrames in kernel
    dataframes_in_kernel = {}
    for name, obj in user_ns.items():
        if isinstance(obj, pd.DataFrame) and not name.startswith('_'):
            dataframes_in_kernel[name] = {
                'shape': obj.shape,
                'columns': list(obj.columns),
                'dtypes': {col: str(dtype) for col, dtype in obj.dtypes.items()}
            }

    # Find DataFrames in DATA_STORE
    dataframes_in_store = {}
    for name, obj in DATA_STORE.items():
        if isinstance(obj, pd.DataFrame):
            dataframes_in_store[name] = {
                'shape': obj.shape,
                'columns': list(obj.columns),
                'dtypes': {col: str(dtype) for col, dtype in obj.dtypes.items()}
            }

    # Combine both
    all_dataframes = {**dataframes_in_store, **dataframes_in_kernel}

    # Find pyrsm model objects in kernel
    models_in_kernel = {}
    for name, obj in user_ns.items():
        if not name.startswith('_'):
            class_name = obj.__class__.__name__
            if class_name in ['regress', 'single_mean', 'compare_means']:
                model_info = {
                    'type': class_name,
                    'variable': name
                }

                # Extract model-specific info
                if class_name == 'regress' and hasattr(obj, 'rvar') and hasattr(obj, 'evar'):
                    model_info['rvar'] = obj.rvar
                    model_info['evar'] = obj.evar if isinstance(obj.evar, list) else [obj.evar]
                    model_info['description'] = f"{obj.rvar} ~ {', '.join(model_info['evar'])}"
                elif class_name == 'single_mean' and hasattr(obj, 'var'):
                    model_info['var'] = obj.var
                    model_info['description'] = f"Single mean test on {obj.var}"
                elif class_name == 'compare_means' and hasattr(obj, 'var1') and hasattr(obj, 'var2'):
                    model_info['var1'] = obj.var1
                    model_info['var2'] = obj.var2
                    model_info['description'] = f"Compare {obj.var1} vs {obj.var2}"

                models_in_kernel[name] = model_info

    # Add models from MODEL_STORE
    models_in_store = {}
    for model_id, info in MODEL_STORE.items():
        models_in_store[model_id] = {
            'type': 'regress',  # Currently only regress models in MODEL_STORE
            'rvar': info['rvar'],
            'evar': info['evar'],
            'description': f"{info['rvar']} ~ {', '.join(info['evar'])}",
            'fitted_at': info['fitted_at']
        }

    # Combine model sources
    all_models = {**models_in_store, **models_in_kernel}

    return {
        'dataframes': all_dataframes,
        'n_dataframes': len(all_dataframes),
        'models': all_models,
        'n_models': len(all_models)
    }


def _build_context_prompt(context):
    """Build context description for LLM"""
    lines = []

    # Add active dataset info
    if ACTIVE_DATASET:
        lines.append(f"🎯 ACTIVE DATASET: {ACTIVE_DATASET}")
        lines.append("   (Use this dataset by default unless user specifies another)")
        lines.append("")

    # Add datasets
    if context.get('dataframes'):
        lines.append("Available datasets:")
        for name, info in context['dataframes'].items():
            active_marker = " ← ACTIVE" if name == ACTIVE_DATASET else ""
            lines.append(f"  - {name}: {info['shape'][0]} rows × {info['shape'][1]} columns{active_marker}")
            lines.append(f"    Columns: {', '.join(info['columns'][:10])}")
            if len(info['columns']) > 10:
                lines.append(f"    ... and {len(info['columns']) - 10} more")
    else:
        lines.append("No datasets currently loaded.")

    # Add models
    if context.get('models'):
        lines.append("\nAvailable models:")
        for name, info in context['models'].items():
            lines.append(f"  - {name} ({info['type']}): {info['description']}")
            if 'variable' in info:
                lines.append(f"    Variable in kernel: {info['variable']}")

    return "\n".join(lines)


def _display_markdown(text):
    """Display text as markdown for proper wrapping"""
    display(Markdown(text))


def _show_dataset_selector(original_prompt):
    """Show interactive dropdown to select active dataset"""
    global ACTIVE_DATASET, LAST_PROMPT

    # Get available datasets
    available_datasets = list(DATA_STORE.keys())

    if not available_datasets:
        display(Markdown("**No datasets loaded yet.**\n\nLoad data first using pyrsm.load_data()"))
        return

    LAST_PROMPT = original_prompt  # Store for re-execution

    # Create widgets
    dropdown = widgets.Dropdown(
        options=available_datasets,
        description='Dataset:',
        style={'description_width': 'initial'}
    )

    button = widgets.Button(
        description='Set Active & Retry',
        button_style='success',
        icon='check'
    )

    output = widgets.Output()

    def on_button_click(b):
        global ACTIVE_DATASET
        selected = dropdown.value
        ACTIVE_DATASET = selected

        with output:
            output.clear_output()
            print(f"✓ Active dataset set to: {ACTIVE_DATASET}")
            print(f"▶ Re-running prompt...")

        # Re-execute the original prompt with active dataset now set
        ipython = get_ipython()
        if ipython and LAST_PROMPT:
            # Execute as %%mcp cell
            ipython.run_cell_magic('mcp', '', LAST_PROMPT)

    button.on_click(on_button_click)

    # Display
    display(Markdown("**Please select a dataset:**"))
    display(widgets.HBox([dropdown, button]))
    display(output)


def _enhance_next_steps(result_text, context):
    """Convert next step suggestions to clickable buttons and markdown"""
    # Find the "Suggested next steps" section
    if "**Suggested next steps:**" not in result_text:
        return result_text, None

    # Split at suggested next steps
    parts = result_text.split("**Suggested next steps:**")
    if len(parts) != 2:
        return result_text, None

    before = parts[0]
    steps_section = parts[1]

    # Parse bullet points
    lines = steps_section.split('\n')
    buttons = []
    markdown_lines = []

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('-'):
            # Remove the dash and extract text
            step_text = line_stripped[1:].strip()

            # Convert to %%mcp prompt
            mcp_prompt = _step_to_prompt(step_text, context)

            # Create button
            button = widgets.Button(
                description=step_text[:50],  # Truncate if too long
                tooltip=mcp_prompt,
                button_style='info',
                layout=widgets.Layout(width='auto', margin='2px')
            )

            # Store prompt as attribute for the click handler
            button.mcp_prompt = mcp_prompt

            def make_click_handler(prompt):
                def on_click(b):
                    ipython = get_ipython()
                    if ipython:
                        # Insert and execute new %%mcp cell
                        ipython.run_cell_magic('mcp', '', prompt)
                return on_click

            button.on_click(make_click_handler(mcp_prompt))
            buttons.append(button)

            # Also keep markdown for reference
            markdown_lines.append(f"- {step_text}")
        elif line_stripped:
            markdown_lines.append(line_stripped)

    # Return both parts
    markdown_result = before + "**Suggested next steps:**\n" + '\n'.join(markdown_lines)

    return markdown_result, buttons


def _step_to_prompt(step_text, context):
    """Convert a next step suggestion to a %%mcp prompt"""
    step_lower = step_text.lower()

    # Get active model name if available
    model_name = None
    if context.get('models'):
        # Use first model as default
        model_name = list(context['models'].keys())[0]

    # Pattern matching for common suggestions
    if 'vif' in step_lower or 'multicollinearity' in step_lower:
        if model_name:
            return f"Calculate VIF scores for {model_name}"
        return "Calculate VIF scores"

    elif 'residual' in step_lower and 'dashboard' in step_lower:
        if model_name:
            return f"Show residual dashboard for {model_name}"
        return "Show residual diagnostics"

    elif 'variable importance' in step_lower or 'permutation' in step_lower:
        if model_name:
            return f"Show variable importance plot for {model_name}"
        return "Show variable importance"

    elif 'plot' in step_lower:
        if model_name:
            return f"Create plots for {model_name}"
        return "Create diagnostic plots"

    # Default: clean up the text
    # Remove parentheticals
    prompt = re.sub(r'\([^)]*\)', '', step_text).strip()
    return prompt


def _show_workflow_selector(analysis_type, user_goal, suggested_data=None):
    """Show interactive widget to choose between guided process or Radiant UI"""

    # Map analysis types to friendly names
    activity_map = {
        'regression_analysis': 'Regression Analysis',
        'hypothesis_testing': 'Hypothesis Testing',
        'compare_groups': 'Compare Groups',
        'explore_data': 'Explore Data',
        'time_series': 'Time Series Analysis',
        'classification': 'Classification',
        'clustering': 'Clustering'
    }

    # Create widgets
    display(Markdown(f"**Analyzing:** {user_goal}"))
    display(Markdown(f"📊 **Recommended Activity:** {activity_map.get(analysis_type, analysis_type)}"))

    mode_radio = widgets.RadioButtons(
        options=[
            ('Guided Process (Learn step-by-step)', 'guided'),
            ('Interactive UI (Work in Radiant app)', 'radiant')
        ],
        description='Mode:',
        value='guided',
        style={'description_width': 'initial'}
    )

    activity_dropdown = widgets.Dropdown(
        options=list(activity_map.values()),
        description='Activity:',
        value=activity_map.get(analysis_type, list(activity_map.values())[0]),
        style={'description_width': 'initial'}
    )

    continue_button = widgets.Button(
        description='Continue',
        button_style='primary',
        icon='arrow-right'
    )

    output = widgets.Output()

    def on_continue(b):
        selected_mode = mode_radio.value
        selected_activity = activity_dropdown.value

        with output:
            output.clear_output()

            if selected_mode == 'guided':
                display(Markdown(f"🎓 **Starting Guided Process** for {selected_activity}..."))
                display(Markdown("_(Guided mode coming soon - for now, running direct analysis)_"))

                # TODO: Implement guided workflow
                # For now, just proceed with regular analysis

            else:  # radiant
                display(Markdown(f"🚀 **Launching Radiant App** for {selected_activity}..."))
                display(Markdown("_(Radiant integration coming soon - for now, running direct analysis)_"))

                # TODO: Launch Radiant app
                # For now, just proceed with regular analysis

    continue_button.on_click(on_continue)

    # Display
    display(mode_radio)
    display(activity_dropdown)
    display(continue_button)
    display(output)


# Define tools in Gemini format
GEMINI_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "single_mean",
                "description": "Perform single-mean hypothesis testing to test if a population mean equals a specific value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_name": {
                            "type": "string",
                            "description": "Name of the loaded dataset",
                        },
                        "var": {
                            "type": "string",
                            "description": "The variable/column name to test",
                        },
                        "comp_value": {
                            "type": "number",
                            "description": "The comparison value for the test",
                        },
                        "alt_hyp": {
                            "type": "string",
                            "description": "Alternative hypothesis: two-sided, greater, or less",
                        },
                        "conf": {
                            "type": "number",
                            "description": "Confidence level (e.g., 0.95)",
                        },
                        "dec": {
                            "type": "integer",
                            "description": "Decimal places in output",
                        }
                    },
                    "required": ["data_name", "var"]
                }
            },
            {
                "name": "compare_means",
                "description": "Compare means between groups using t-test or Wilcoxon test",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_name": {
                            "type": "string",
                            "description": "Name of the loaded dataset",
                        },
                        "var1": {
                            "type": "string",
                            "description": "First variable (categorical for grouping)",
                        },
                        "var2": {
                            "type": "string",
                            "description": "Second variable (numeric to compare)",
                        },
                        "test_type": {
                            "type": "string",
                            "description": "Test type: t-test or wilcox",
                        },
                        "dec": {
                            "type": "integer",
                            "description": "Decimal places in output",
                        }
                    },
                    "required": ["data_name", "var1", "var2"]
                }
            },
            {
                "name": "regress_fit",
                "description": "Fit a linear regression model to predict a response variable from explanatory variables",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_name": {
                            "type": "string",
                            "description": "Name of the loaded dataset",
                        },
                        "rvar": {
                            "type": "string",
                            "description": "Response (dependent) variable",
                        },
                        "evar": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of explanatory (independent) variables",
                        },
                        "ivar": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Interaction terms (e.g., ['x1:x2'])",
                        },
                        "vif": {
                            "type": "boolean",
                            "description": "Include VIF to check multicollinearity",
                        },
                        "dec": {
                            "type": "integer",
                            "description": "Decimal places in output",
                        }
                    },
                    "required": ["data_name", "rvar", "evar"]
                }
            },
            {
                "name": "model_analyze",
                "description": "Perform additional analysis on an EXISTING model object (regression, single_mean, compare_means) that's already in the kernel. Use this when the user refers to a model by name (e.g., 'reg', 'sm').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model variable in the kernel (e.g., 'reg', 'sm', 'cm')",
                        },
                        "operation": {
                            "type": "string",
                            "description": "Operation to perform: 'summary' (show summary with options), 'vif' (variance inflation factors), 'plot' (diagnostic plots), 'predict' (predictions)",
                        },
                        "vif": {
                            "type": "boolean",
                            "description": "For summary: include VIF scores",
                        },
                        "plot_type": {
                            "type": "string",
                            "description": "For plot operation: 'dashboard', 'vimp', 'pred', 'residual', 'hist', 'box', 'density'",
                        },
                        "dec": {
                            "type": "integer",
                            "description": "Decimal places in output",
                        }
                    },
                    "required": ["model_name", "operation"]
                }
            },
            {
                "name": "workflow_selector",
                "description": "Ask user to choose between guided learning process or interactive Radiant UI for their analysis. Use this when user requests analysis that could benefit from either approach.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis: regression_analysis, hypothesis_testing, compare_groups, explore_data, time_series, classification, clustering",
                        },
                        "user_goal": {
                            "type": "string",
                            "description": "What the user wants to accomplish (in their words)",
                        },
                        "suggested_data": {
                            "type": "string",
                            "description": "Dataset to use (if known)",
                        }
                    },
                    "required": ["analysis_type", "user_goal"]
                }
            }
        ]
    }
]


@register_cell_magic
def mcp(line, cell):
    """
    MCP bridge: natural language -> LLM tool calling -> MCP tools -> execution

    Example:
        %%mcp
        Compare salary between academic ranks
    """
    ipython = get_ipython()

    # 1. Get context
    context = _get_kernel_context()
    context_prompt = _build_context_prompt(context)

    if context['n_dataframes'] == 0:
        print("⚠ No datasets loaded. Load data first with:")
        print("  data, _ = pyrsm.load_data(name='diamonds', pkg='model')")
        print("  DATA_STORE['diamonds'] = data")
        return

    # 2. Load API key and create model
    try:
        _load_gemini_api_key()
        model = genai.GenerativeModel('gemini-2.0-flash-exp', tools=GEMINI_TOOLS)
    except Exception as e:
        print(f"✗ Error setting up Gemini: {e}")
        return

    # 3. Build full prompt with instructions
    user_prompt = cell.strip()

    # Build system instructions with active dataset info
    system_instructions = """You are a data analysis assistant.

IMPORTANT: When handling user requests, follow these rules:

DATASET SELECTION:
- If you see "🎯 ACTIVE DATASET", use that dataset BY DEFAULT for all operations
- Only use a different dataset if the user explicitly names it in their request
- If no active dataset is set and multiple datasets exist, ask which to use

WORKFLOW SELECTION:
- For complex analysis requests (regression, hypothesis testing, comparisons), consider offering the workflow_selector
- Use workflow_selector when user could benefit from either guided learning OR interactive Radiant UI
- Examples that should trigger workflow_selector:
  - "I want to understand what drives diamond prices" → workflow_selector
  - "Analyze the relationship between X and Y" → workflow_selector
  - "Compare groups" → workflow_selector
- DON'T use workflow_selector for simple, specific tasks:
  - "Calculate VIF for reg" → use model_analyze directly
  - "Show mean salary" → use single_mean directly

MODEL REUSE:
- ALWAYS check if a relevant model already exists in "Available models" section
- If a model exists that the user is referring to, use the model_analyze tool to work with it
- Only create new models when explicitly asked or when no suitable model exists
- Examples:
  - "Calculate VIF for reg" → use model_analyze tool with model_name="reg", operation="vif"
  - "Show dashboard plot for reg" → use model_analyze tool with model_name="reg", operation="plot", plot_type="dashboard"
  - "Get summary of sm model" → use model_analyze tool with model_name="sm", operation="summary"

The model_analyze tool lets you work with existing models without refitting them."""

    full_prompt = f"{system_instructions}\n\n{context_prompt}\n\nUser request: {user_prompt}"

    print(f"🤖 Processing: {user_prompt}")
    print(f"📊 Context: {context['n_dataframes']} dataset(s), {context.get('n_models', 0)} model(s) available")
    print()

    # 4. Call LLM
    try:
        response = model.generate_content(full_prompt)
    except Exception as e:
        print(f"✗ Error calling LLM: {e}")
        return

    # 5. Extract tool calls
    tool_calls = []
    if response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                args = part.function_call.args
                tool_calls.append({
                    'name': part.function_call.name,
                    'arguments': dict(args) if args else {}
                })

    if not tool_calls:
        # LLM returned text without tool call
        if hasattr(response, 'text'):
            response_text = response.text

            # Check if LLM is asking about dataset selection
            if any(phrase in response_text.lower() for phrase in ['which dataset', 'select dataset', 'what dataset', 'choose dataset']):
                # Show interactive dataset selector
                _display_markdown(response_text)
                _show_dataset_selector(user_prompt)
            else:
                # Just display as markdown for proper wrapping
                _display_markdown(response_text)
        else:
            display(Markdown("✗ LLM did not generate a tool call or text response"))
        return

    # 6. Execute tool calls via MCP
    for tool_call in tool_calls:
        print(f"🔧 Tool: {tool_call['name']}")
        print(f"   Args: {tool_call['arguments']}")
        print()

        # Special handling for workflow_selector - shows interactive widget
        if tool_call['name'] == 'workflow_selector':
            analysis_type = tool_call['arguments']['analysis_type']
            user_goal = tool_call['arguments']['user_goal']
            suggested_data = tool_call['arguments'].get('suggested_data')

            _show_workflow_selector(analysis_type, user_goal, suggested_data)
            return  # Wait for user selection

        # Special handling for model_analyze - operates on kernel objects directly
        elif tool_call['name'] == 'model_analyze':
            model_name = tool_call['arguments']['model_name']
            operation = tool_call['arguments']['operation']

            # Get model from kernel
            user_ns = ipython.user_ns
            if model_name not in user_ns:
                print(f"✗ Model '{model_name}' not found in kernel namespace")
                continue

            model_obj = user_ns[model_name]
            model_type = model_obj.__class__.__name__

            # Generate code based on operation
            if operation == 'vif' or (operation == 'summary' and tool_call['arguments'].get('vif')):
                generated_code = f"{model_name}.summary(vif=True)"
            elif operation == 'summary':
                dec = tool_call['arguments'].get('dec', 3)
                generated_code = f"{model_name}.summary(dec={dec})"
            elif operation == 'plot':
                plot_type = tool_call['arguments'].get('plot_type', 'dashboard')
                generated_code = f"{model_name}.plot(plots='{plot_type}')"
            else:
                print(f"✗ Unknown operation: {operation}")
                continue

            # Insert and execute
            ipython.set_next_input(generated_code, replace=False)
            print("▶ Executing...")
            print()
            ipython.run_cell(generated_code)
            continue

        # Call MCP tool for other tools
        async def run_tool():
            return await call_tool(
                name=tool_call['name'],
                arguments=tool_call['arguments']
            )

        try:
            result = asyncio.run(run_tool())
            result_text = result[0].text

            # Extract generated code
            if "```python" in result_text:
                code_start = result_text.index("```python") + 9
                code_end = result_text.index("```", code_start)
                generated_code = result_text[code_start:code_end].strip()

                # 7. Insert code in next cell
                ipython.set_next_input(generated_code, replace=False)

                # 8. Auto-execute
                print("▶ Executing generated code...")
                print()
                ipython.run_cell(generated_code)
            else:
                # No code to execute, enhance and show result
                enhanced_text, buttons = _enhance_next_steps(result_text, context)
                _display_markdown(enhanced_text)

                # Show interactive buttons if available
                if buttons:
                    display(Markdown("**Quick actions:**"))
                    display(widgets.VBox(buttons, layout=widgets.Layout(gap='5px')))

        except Exception as e:
            print(f"✗ Error executing tool: {e}")
            return


@register_line_magic
def mcp_info(line):
    """Show current context (loaded datasets and models)"""
    context = _get_kernel_context()
    print(_build_context_prompt(context))


@register_line_magic
def mcp_use(line):
    """Set the active dataset for analysis

    Usage:
        %mcp_use salary
        %mcp_use diamonds
    """
    global ACTIVE_DATASET

    dataset_name = line.strip()

    if not dataset_name:
        # Show current active dataset
        if ACTIVE_DATASET:
            print(f"🎯 Current active dataset: {ACTIVE_DATASET}")
        else:
            print("No active dataset set.")
            print("\nAvailable datasets:")
            for name in DATA_STORE.keys():
                print(f"  - {name}")
            print("\nUsage: %mcp_use <dataset_name>")
        return

    # Check if dataset exists
    ipython = get_ipython()
    user_ns = ipython.user_ns if ipython else {}

    # Check both kernel namespace and DATA_STORE
    if dataset_name in DATA_STORE or (dataset_name in user_ns and isinstance(user_ns[dataset_name], pd.DataFrame)):
        ACTIVE_DATASET = dataset_name
        print(f"✓ Active dataset set to: {ACTIVE_DATASET}")

        # Show info about the dataset
        if dataset_name in DATA_STORE:
            df = DATA_STORE[dataset_name]
        else:
            df = user_ns[dataset_name]

        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns: {', '.join(list(df.columns)[:10])}")
        if len(df.columns) > 10:
            print(f"           ... and {len(df.columns) - 10} more")
    else:
        print(f"✗ Dataset '{dataset_name}' not found")
        print("\nAvailable datasets:")
        for name in DATA_STORE.keys():
            print(f"  - {name}")
        if user_ns:
            for name, obj in user_ns.items():
                if isinstance(obj, pd.DataFrame) and not name.startswith('_') and name not in DATA_STORE:
                    print(f"  - {name} (in kernel)")



def load_ipython_extension(ipython):
    """Load the extension"""
    print("✓ MCP Bridge Magic loaded!")
    print()
    print("Usage:")
    print("  %%mcp")
    print("  Your natural language request here")
    print()
    print("Commands:")
    print("  %mcp_info       - Show available datasets and models")
    print("  %mcp_use <name> - Set active dataset (default for all operations)")
    print()
    print("Example:")
    print("  %mcp_use salary")
    print("  %%mcp")
    print("  Test if mean salary equals 100000")


def unload_ipython_extension(ipython):
    """Unload the extension"""
    pass
