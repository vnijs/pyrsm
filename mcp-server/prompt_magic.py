"""
IPython magic extension for generating pyrsm code from natural language prompts.

Usage:
    %load_ext prompt_magic

    %%prompt
    Fit a regression with price as response and carat, depth as predictors
"""

from IPython.core.magic import register_cell_magic, register_line_magic
from IPython import get_ipython
import pandas as pd
import sys
import json


def _get_kernel_context():
    """Extract context from current IPython kernel namespace."""
    ipython = get_ipython()
    if ipython is None:
        return {}

    user_ns = ipython.user_ns

    # Find DataFrames
    dataframes = {}
    for name, obj in user_ns.items():
        if isinstance(obj, pd.DataFrame) and not name.startswith('_'):
            dataframes[name] = {
                'shape': obj.shape,
                'columns': list(obj.columns),
                'dtypes': {col: str(dtype) for col, dtype in obj.dtypes.items()}
            }

    # Find pyrsm model objects (they have .summary() method)
    models = {}
    for name, obj in user_ns.items():
        if not name.startswith('_') and hasattr(obj, 'summary') and hasattr(obj, 'evar'):
            models[name] = {
                'type': type(obj).__name__,
                'has_summary': True
            }

    # Get imported modules
    imported_modules = [m for m in sys.modules.keys()
                       if not m.startswith('_') and '.' not in m]

    # Check if pyrsm is imported
    has_pyrsm = 'pyrsm' in imported_modules

    return {
        'dataframes': dataframes,
        'models': models,
        'has_pyrsm': has_pyrsm,
        'n_dataframes': len(dataframes),
        'n_models': len(models)
    }


def _generate_code_from_prompt(prompt, context):
    """
    Generate pyrsm code from natural language prompt.

    For prototype: Simple rule-based generation.
    Later: Call MCP server's code_generate tool.
    """
    prompt_lower = prompt.lower().strip()

    # Simple pattern matching for prototype
    generated_code = []
    comments = []

    # Check if pyrsm needs to be imported
    if not context.get('has_pyrsm'):
        generated_code.append("import pyrsm")
        comments.append("# Import pyrsm")

    # Pattern: Load data
    if 'load' in prompt_lower and 'data' in prompt_lower:
        if 'diamonds' in prompt_lower:
            comments.append("# Load diamonds dataset")
            generated_code.append("diamonds, desc = pyrsm.load_data(name='diamonds', pkg='model')")
            generated_code.append("print(f'Loaded: {diamonds.shape[0]} rows × {diamonds.shape[1]} columns')")
        else:
            comments.append("# Load dataset")
            generated_code.append("# TODO: Specify dataset name")
            generated_code.append("data, desc = pyrsm.load_data('dataset_name', 'package_name')")

    # Pattern: Regression
    elif 'regression' in prompt_lower or 'regress' in prompt_lower:
        # Find available DataFrames
        df_names = list(context.get('dataframes', {}).keys())

        if df_names:
            df_name = df_names[0]  # Use first available DataFrame
            df_info = context['dataframes'][df_name]

            # Try to extract variable names from prompt
            rvar = None
            evar = []

            # Look for common response variable patterns
            if 'price' in prompt_lower:
                rvar = 'price'
            elif 'sales' in prompt_lower:
                rvar = 'sales'

            # Look for predictor patterns
            if 'carat' in prompt_lower:
                evar.append('carat')
            if 'depth' in prompt_lower:
                evar.append('depth')
            if 'table' in prompt_lower:
                evar.append('table')

            # Default to first few columns if not specified
            if not rvar and df_info['columns']:
                rvar = df_info['columns'][0]
            if not evar and len(df_info['columns']) > 1:
                evar = df_info['columns'][1:min(4, len(df_info['columns']))]

            comments.append(f"# Regression: {rvar} ~ {' + '.join(evar)}")
            generated_code.append(f"reg = pyrsm.model.regress({{'{df_name}': {df_name}}}, rvar='{rvar}', evar={evar})")
            generated_code.append("reg.summary()")

        else:
            comments.append("# No DataFrame found - load data first")
            generated_code.append("# Please load a dataset first using pyrsm.load_data()")

    # Pattern: Summary statistics
    elif 'summary' in prompt_lower or 'describe' in prompt_lower or 'statistics' in prompt_lower:
        df_names = list(context.get('dataframes', {}).keys())
        if df_names:
            df_name = df_names[0]
            comments.append(f"# Summary statistics for {df_name}")
            generated_code.append(f"{df_name}.describe()")
        else:
            comments.append("# No DataFrame found")
            generated_code.append("# Load data first")

    # Pattern: Compare means
    elif 'compare' in prompt_lower and 'mean' in prompt_lower:
        df_names = list(context.get('dataframes', {}).keys())
        if df_names:
            df_name = df_names[0]
            comments.append("# Compare means")
            generated_code.append("# TODO: Specify var1 and var2")
            generated_code.append(f"result = pyrsm.basics.compare_means({df_name}, var1='group_a', var2='group_b')")
            generated_code.append("result.summary()")
        else:
            generated_code.append("# Load data first")

    # Default: provide helpful comment
    else:
        comments.append("# Generated from prompt (pattern not recognized)")
        comments.append(f"# Prompt: {prompt}")
        generated_code.append("# Available DataFrames: " + ", ".join(context.get('dataframes', {}).keys()))
        generated_code.append("# Available models: " + ", ".join(context.get('models', {}).keys()))
        generated_code.append("# Try: 'fit regression', 'load data', 'summary statistics'")

    # Combine comments and code
    result = "\n".join(comments) + "\n" + "\n".join(generated_code) if comments else "\n".join(generated_code)
    return result


@register_cell_magic
def prompt(line, cell):
    """
    Generate and execute pyrsm code from natural language prompt.

    Example:
        %%prompt
        Fit a regression with price as response and carat, depth as predictors
    """
    ipython = get_ipython()

    # Get context from kernel
    context = _get_kernel_context()

    # Generate code
    prompt_text = cell.strip()
    generated_code = _generate_code_from_prompt(prompt_text, context)

    # Insert code in next cell
    ipython.set_next_input(generated_code, replace=False)

    # Auto-execute the generated code
    result = ipython.run_cell(generated_code)

    # Return None to suppress ExecutionResult display
    return None


@register_line_magic
def prompt_line(line):
    """Single-line version of %%prompt"""
    ipython = get_ipython()
    context = _get_kernel_context()
    generated_code = _generate_code_from_prompt(line, context)

    ipython.set_next_input(generated_code, replace=False)
    result = ipython.run_cell(generated_code)
    return result


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    print("✓ Prompt magic loaded!")
    print("  Usage: %%prompt")
    print("  Example:")
    print("    %%prompt")
    print("    Fit a regression with price as response")


def unload_ipython_extension(ipython):
    """Unload the extension."""
    pass
