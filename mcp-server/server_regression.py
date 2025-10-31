#!/usr/bin/env python3
"""
pyrsm MCP Server - Regression Tools with State Management

Exposes pyrsm.model.regress as MCP tools with persistent model storage.
"""

import asyncio
import sys
import hashlib
import time
from datetime import datetime
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import pandas as pd
import pyrsm
from io import StringIO
from contextlib import redirect_stdout

# Create server
app = Server("pyrsm-regression-mcp")

# Model registry - stores fitted models for reuse
MODEL_STORE = {}

# Data registry - stores loaded datasets for reuse
DATA_STORE = {
    'sample': pd.DataFrame({
        'sales': [100, 150, 200, 250, 300, 120, 180, 220, 280, 320, 95, 145, 195, 245, 295],
        'x1': [1, 2, 3, 4, 5, 1.5, 2.5, 3.5, 4.5, 5.5, 1.2, 2.2, 3.2, 4.2, 5.2],
        'x2': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55, 12, 22, 32, 42, 52],
        'x3': [5, 10, 15, 20, 25, 7, 12, 17, 22, 27, 6, 11, 16, 21, 26],
        'price': [95, 105, 98, 102, 110, 97, 103, 99, 101, 108, 96, 104, 100, 103, 107],
    })
}

# Data descriptions
DATA_DESCRIPTIONS = {
    'sample': 'Sample dataset with sales, price, and predictor variables (x1, x2, x3). 15 observations.'
}

def generate_model_id(rvar: str, evar: list, timestamp: float) -> str:
    """Generate unique model ID"""
    vars_str = f"{rvar}_{'_'.join(sorted(evar))}"
    hash_obj = hashlib.md5(vars_str.encode())
    return f"reg_{hash_obj.hexdigest()[:8]}_{int(timestamp)}"

def store_model(model_obj, rvar: str, evar: list, **kwargs) -> str:
    """Store model and return ID"""
    timestamp = time.time()
    model_id = generate_model_id(rvar, evar, timestamp)

    MODEL_STORE[model_id] = {
        'obj': model_obj,
        'rvar': rvar,
        'evar': evar,
        'fitted_at': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
        'metadata': kwargs
    }

    return model_id

def get_model(model_id: str):
    """Retrieve stored model"""
    if model_id not in MODEL_STORE:
        raise ValueError(f"Model '{model_id}' not found. Available: {list(MODEL_STORE.keys())}")
    return MODEL_STORE[model_id]

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available regression tools"""
    return [
        Tool(
            name="data_load",
            description="Load a dataset from pyrsm's built-in examples or list available datasets",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Dataset name (e.g., 'diamonds', 'salary', 'titanic'). Leave empty to list available datasets.",
                    },
                    "package": {
                        "type": "string",
                        "enum": ["basics", "model", "data", "design", "multivariate"],
                        "description": "Package category. Leave empty to search all packages.",
                    }
                },
            }
        ),
        Tool(
            name="data_load_file",
            description="Load data from a CSV, Excel, Parquet, or JSON file on your system",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the data file (e.g., '/home/vnijs/data/mydata.csv')",
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Name to store this dataset as (defaults to filename without extension)",
                    },
                    "file_type": {
                        "type": "string",
                        "enum": ["csv", "excel", "parquet", "json", "auto"],
                        "description": "File type (auto-detected from extension if not specified)",
                        "default": "auto"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="data_list",
            description="List all currently loaded datasets",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="data_info",
            description="Get information about a loaded dataset (columns, types, shape, description)",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_name": {
                        "type": "string",
                        "description": "Name of the loaded dataset",
                    }
                },
                "required": ["data_name"]
            }
        ),
        Tool(
            name="regress_fit",
            description="Fit a linear regression model and store it for later use. Returns model ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_name": {
                        "type": "string",
                        "description": "Name of loaded dataset to use (from data_load or data_list)",
                        "default": "sample"
                    },
                    "rvar": {
                        "type": "string",
                        "description": "Response (dependent) variable name",
                    },
                    "evar": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of explanatory (independent) variable names",
                    },
                    "ivar": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of interaction terms (e.g., ['x1:x2', 'x2:x3']). Format: 'var1:var2'",
                        "default": []
                    },
                    "show_summary": {
                        "type": "boolean",
                        "description": "Show summary output immediately",
                        "default": True
                    },
                    "vif": {
                        "type": "boolean",
                        "description": "Include VIF (variance inflation factor) to check multicollinearity",
                        "default": False
                    },
                    "dec": {
                        "type": "integer",
                        "description": "Number of decimal places to display in summary",
                        "default": 3
                    },
                    "plots": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["dashboard", "vimp"]},
                        "description": "Plot types: 'dashboard' for residual diagnostics, 'vimp' for variable importance",
                        "default": []
                    }
                },
                "required": ["rvar", "evar"]
            }
        ),
        Tool(
            name="regress_summary",
            description="Get summary statistics from a fitted regression model (no refitting).",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID from regress_fit",
                    },
                    "vif": {
                        "type": "boolean",
                        "description": "Show VIF (variance inflation factors)",
                        "default": False
                    },
                    "fit": {
                        "type": "boolean",
                        "description": "Show model fit statistics",
                        "default": True
                    },
                    "dec": {
                        "type": "integer",
                        "description": "Decimal places",
                        "default": 3
                    }
                },
                "required": ["model_id"]
            }
        ),
        Tool(
            name="regress_plot",
            description="Generate diagnostic plots from a fitted regression model (no refitting).",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID from regress_fit",
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["dashboard", "vimp", "pred"],
                        "description": "Type of plot: 'dashboard' (residuals), 'vimp' (variable importance), 'pred' (predictions)",
                    }
                },
                "required": ["model_id", "plot_type"]
            }
        ),
        Tool(
            name="regress_list",
            description="List all stored regression models",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="single_mean",
            description="Perform single-mean hypothesis testing to test if a population mean equals a specific value",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_name": {
                        "type": "string",
                        "description": "Name of the loaded dataset to use for hypothesis testing",
                    },
                    "var": {
                        "type": "string",
                        "description": "The variable/column name to test (must be numeric)",
                    },
                    "alt_hyp": {
                        "type": "string",
                        "enum": ["two-sided", "greater", "less"],
                        "description": "The alternative hypothesis",
                        "default": "two-sided"
                    },
                    "conf": {
                        "type": "number",
                        "description": "The confidence level for the test (e.g., 0.95 for 95%)",
                        "default": 0.95
                    },
                    "comp_value": {
                        "type": "number",
                        "description": "The comparison value for the test (value under null hypothesis)",
                        "default": 0
                    },
                    "dec": {
                        "type": "integer",
                        "description": "Number of decimals to show in summary output",
                        "default": 3
                    },
                    "plots": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["hist", "sim"]},
                        "description": "Plots to generate: 'hist' for histogram with confidence interval, 'sim' for simulation",
                        "default": []
                    }
                },
                "required": ["data_name", "var"]
            }
        ),
        Tool(
            name="compare_means",
            description="Compare means between groups using t-test or Wilcoxon test to determine if there are significant differences",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_name": {
                        "type": "string",
                        "description": "Name of the loaded dataset to use for hypothesis testing",
                    },
                    "var1": {
                        "type": "string",
                        "description": "First variable/column name (can be numeric or categorical for grouping)",
                    },
                    "var2": {
                        "type": "string",
                        "description": "Second variable/column name (must be numeric - this is what we're comparing)",
                    },
                    "alt_hyp": {
                        "type": "string",
                        "enum": ["two-sided", "greater", "less"],
                        "description": "The alternative hypothesis",
                        "default": "two-sided"
                    },
                    "conf": {
                        "type": "number",
                        "description": "The confidence level for the test",
                        "default": 0.95
                    },
                    "sample_type": {
                        "type": "string",
                        "enum": ["independent", "paired"],
                        "description": "Type of samples: 'independent' for different groups, 'paired' for before/after",
                        "default": "independent"
                    },
                    "test_type": {
                        "type": "string",
                        "enum": ["t-test", "wilcox"],
                        "description": "Type of test: 't-test' (parametric) or 'wilcox' (non-parametric)",
                        "default": "t-test"
                    },
                    "dec": {
                        "type": "integer",
                        "description": "Number of decimals to show in summary output",
                        "default": 3
                    },
                    "plots": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["scatter", "box", "density", "bar"]},
                        "description": "Plot types: 'scatter', 'box', 'density', 'bar'",
                        "default": []
                    }
                },
                "required": ["data_name", "var1", "var2"]
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "data_load_file":
        file_path = arguments["file_path"]
        dataset_name = arguments.get("dataset_name")
        file_type = arguments.get("file_type", "auto")

        # Auto-detect dataset name from filename
        if not dataset_name:
            import os
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]

        # Auto-detect file type from extension
        if file_type == "auto":
            if file_path.endswith('.csv'):
                file_type = "csv"
            elif file_path.endswith(('.xlsx', '.xls')):
                file_type = "excel"
            elif file_path.endswith('.parquet'):
                file_type = "parquet"
            elif file_path.endswith('.json'):
                file_type = "json"
            else:
                return [TextContent(type="text", text=f"Cannot auto-detect file type from: {file_path}. Please specify file_type.")]

        # Load the file
        try:
            if file_type == "csv":
                data = pd.read_csv(file_path)
                code = f"data = pd.read_csv('{file_path}')"
            elif file_type == "excel":
                data = pd.read_excel(file_path)
                code = f"data = pd.read_excel('{file_path}')"
            elif file_type == "parquet":
                data = pd.read_parquet(file_path)
                code = f"data = pd.read_parquet('{file_path}')"
            elif file_type == "json":
                data = pd.read_json(file_path)
                code = f"data = pd.read_json('{file_path}')"
            else:
                return [TextContent(type="text", text=f"Unsupported file type: {file_type}")]

            # Store in registry
            DATA_STORE[dataset_name] = data
            DATA_DESCRIPTIONS[dataset_name] = f"Loaded from {file_path}"

            result = f"✓ File loaded: {dataset_name}\n\n"
            result += f"Source: {file_path}\n"
            result += f"Shape: {data.shape[0]} rows × {data.shape[1]} columns\n"
            result += f"Columns: {', '.join(list(data.columns))}\n\n"
            result += f"Code:\n```python\nimport pandas as pd\n{code}\n```\n\n"
            result += f"First few rows:\n```\n{data.head(3).to_string()}\n```\n\n"
            result += "**Suggested next steps:**\n"
            result += f"- Explore data types and missing values\n"
            result += f"- Run regression analysis using this dataset\n"
            result += f"- Check data summary statistics\n"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error loading file: {str(e)}\n\nPlease check:\n- File path is correct\n- File format matches type\n- File is not corrupted")]

    elif name == "data_load":
        dataset_name = arguments.get("name")
        package = arguments.get("package")

        # If no name provided, list available datasets
        if not dataset_name:
            try:
                # Get all available datasets from pyrsm
                all_data, all_desc = pyrsm.load_data(pkg=package)
                result = f"Available datasets{' in ' + package if package else ''}:\n\n"
                for name in sorted(all_data.keys()):
                    desc_preview = all_desc[name].split('\n')[0][:80]
                    result += f"• {name} - {desc_preview}...\n"
                result += f"\n💡 Use data_load(name='dataset_name') to load a specific dataset"
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error listing datasets: {str(e)}")]

        # Load specific dataset
        try:
            data, description = pyrsm.load_data(pkg=package, name=dataset_name)

            # Store in registry
            DATA_STORE[dataset_name] = data
            DATA_DESCRIPTIONS[dataset_name] = description

            # Get info
            shape = data.shape
            columns = list(data.columns)
            dtypes = data.dtypes.to_dict()

            result = f"✓ Dataset loaded: {dataset_name}\n\n"
            result += f"Shape: {shape[0]} rows × {shape[1]} columns\n"
            result += f"Columns: {', '.join(columns)}\n\n"
            result += f"Description:\n{description[:200]}...\n\n"
            result += f"Code:\n```python\n"
            result += f"data, description = pyrsm.load_data("
            if package:
                result += f"pkg='{package}', "
            result += f"name='{dataset_name}')\n```\n\n"
            result += f"💡 Use data_info(data_name='{dataset_name}') for full details\n\n"

            # Add suggested next steps
            result += "**Suggested next steps:**\n"
            result += f"- Explore the data structure and variables\n"
            result += f"- Run a regression analysis on this dataset\n"
            result += f"- Ask me to explain what this dataset contains\n"

            return [TextContent(type="text", text=result)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error loading {dataset_name}: {str(e)}")]

    elif name == "data_list":
        if not DATA_STORE:
            return [TextContent(type="text", text="No datasets loaded yet. Use data_load() to load a dataset.")]

        result = f"Loaded datasets ({len(DATA_STORE)}):\n\n"
        for name, data in DATA_STORE.items():
            result += f"• {name}: {data.shape[0]} rows × {data.shape[1]} columns\n"
            result += f"  Columns: {', '.join(list(data.columns)[:5])}"
            if len(data.columns) > 5:
                result += f", ... ({len(data.columns)} total)"
            result += "\n\n"

        return [TextContent(type="text", text=result)]

    elif name == "data_info":
        data_name = arguments["data_name"]

        if data_name not in DATA_STORE:
            return [TextContent(type="text", text=f"Dataset '{data_name}' not found. Use data_list() to see loaded datasets.")]

        data = DATA_STORE[data_name]
        desc = DATA_DESCRIPTIONS.get(data_name, "No description available")

        result = f"Dataset: {data_name}\n"
        result += f"{'='*60}\n\n"
        result += f"Shape: {data.shape[0]} rows × {data.shape[1]} columns\n\n"
        result += f"Columns and types:\n"
        for col, dtype in data.dtypes.items():
            result += f"  • {col}: {dtype}\n"
        result += f"\nFirst few rows:\n```\n{data.head(3).to_string()}\n```\n\n"
        result += f"Description:\n{desc}\n"

        return [TextContent(type="text", text=result)]

    elif name == "regress_fit":
        data_name = arguments.get("data_name", "sample")
        rvar = arguments["rvar"]
        evar = arguments["evar"]
        show_summary = arguments.get("show_summary", True)
        vif = arguments.get("vif", False)

        # Get the dataset
        if data_name not in DATA_STORE:
            return [TextContent(type="text", text=f"Dataset '{data_name}' not found. Use data_load() or data_list() first.")]

        data = DATA_STORE[data_name]

        # Fit the model using pyrsm
        reg = pyrsm.model.regress(data, rvar=rvar, evar=evar)

        # Store the model
        model_id = store_model(reg, rvar, evar, data_name=data_name)

        # Generate code
        evar_str = str(evar)
        code = f"data, description = pyrsm.load_data(name='{data_name}')\n"
        code += f"reg = pyrsm.model.regress(data, rvar='{rvar}', evar={evar_str})\n"
        code += f"reg.summary(vif={vif})"

        result = f"✓ Model fitted and stored as: {model_id}\n\n"
        result += f"Dataset: {data_name} ({data.shape[0]} rows)\n"
        result += f"Response: {rvar}\n"
        result += f"Predictors: {', '.join(evar)}\n\n"
        result += f"Generated code:\n```python\n{code}\n```\n\n"

        if show_summary:
            output = StringIO()
            with redirect_stdout(output):
                reg.summary(vif=vif)

            result += f"Summary:\n```\n{output.getvalue()}```\n\n"

        result += f"💡 Use model_id '{model_id}' for:\n"
        result += "  • regress_summary - Get additional statistics\n"
        result += "  • regress_plot - Generate diagnostic plots\n\n"

        # Add suggested next steps
        result += "**Suggested next steps:**\n"
        result += f"- Check for multicollinearity (VIF)\n"
        result += f"- View residual diagnostics (dashboard plot)\n"
        result += f"- Check variable importance (permutation importance)\n"
        result += f"- Interpret the coefficients and p-values\n"

        return [TextContent(type="text", text=result)]

    elif name == "regress_summary":
        model_id = arguments["model_id"]
        vif = arguments.get("vif", False)
        fit = arguments.get("fit", True)
        dec = arguments.get("dec", 3)

        # Get stored model (no refitting!)
        model_info = get_model(model_id)
        reg = model_info['obj']

        # Get summary
        output = StringIO()
        with redirect_stdout(output):
            reg.summary(vif=vif, fit=fit, dec=dec)

        code = f"reg.summary(vif={vif}, fit={fit}, dec={dec})"

        result = f"Summary for model: {model_id}\n"
        result += f"(fitted at {model_info['fitted_at']})\n\n"
        result += f"Code:\n```python\n{code}\n```\n\n"
        result += f"Output:\n```\n{output.getvalue()}```\n\n"

        # Add suggested next steps
        result += "**Suggested next steps:**\n"
        if not vif:
            result += f"- Check for multicollinearity (VIF)\n"
        result += f"- View residual diagnostics (dashboard plot)\n"
        result += f"- Check variable importance (permutation importance)\n"
        result += f"- Interpret the statistical significance of coefficients\n"

        return [TextContent(type="text", text=result)]

    elif name == "regress_plot":
        model_id = arguments["model_id"]
        plot_type = arguments["plot_type"]

        # Get stored model (no refitting!)
        model_info = get_model(model_id)
        reg = model_info['obj']

        result = f"Plot type '{plot_type}' for model: {model_id}\n"
        result += f"(fitted at {model_info['fitted_at']})\n\n"
        result += f"Code:\n```python\nreg.plot(plots='{plot_type}')\n```\n\n"
        result += "📊 In a full implementation, the plot would be rendered here.\n"
        result += "   For now, run the code above in your environment to see the plot.\n\n"

        # Add suggested next steps based on plot type
        result += "**Suggested next steps:**\n"
        if plot_type == 'dashboard':
            result += "- Interpret residual patterns (look for non-linearity, heteroscedasticity)\n"
            result += "- Check for influential observations\n"
            result += "- Try variable importance plot if model fits well\n"
        elif plot_type == 'vimp':
            result += "- Identify which variables matter most\n"
            result += "- Consider simplifying model by removing unimportant variables\n"
            result += "- Check residual diagnostics if you haven't already\n"
        else:
            result += "- View other diagnostic plots\n"
            result += "- Interpret the model results\n"

        return [TextContent(type="text", text=result)]

    elif name == "regress_list":
        if not MODEL_STORE:
            return [TextContent(type="text", text="No models stored yet.")]

        result = f"Stored models ({len(MODEL_STORE)}):\n\n"
        for model_id, info in MODEL_STORE.items():
            result += f"• {model_id}\n"
            result += f"  Response: {info['rvar']}\n"
            result += f"  Predictors: {', '.join(info['evar'])}\n"
            result += f"  Fitted: {info['fitted_at']}\n\n"

        return [TextContent(type="text", text=result)]

    if name == "code_generate":
        prompt = arguments["prompt"]
        context = arguments.get("context", {})

        # For prototype: Use simple pattern matching (same logic as prompt_magic.py)
        # Future: Integrate with LLM for smarter generation

        prompt_lower = prompt.lower().strip()
        generated_code = []
        comments = []

        # Get available dataframes and models from context or DATA_STORE
        dataframes = context.get('dataframes', {})
        if not dataframes:
            # Use DATA_STORE as fallback
            dataframes = {name: {'shape': (len(df), len(df.columns)), 'columns': list(df.columns)}
                         for name, df in DATA_STORE.items()}

        # Pattern: Load data
        if 'load' in prompt_lower and 'data' in prompt_lower:
            if 'diamonds' in prompt_lower:
                comments.append("# Load diamonds dataset")
                generated_code.append("diamonds, desc = pyrsm.load_data(name='diamonds', pkg='model')")
                generated_code.append("print(f'Loaded: {diamonds.shape[0]} rows × {diamonds.shape[1]} columns')")
            else:
                comments.append("# Load dataset - specify name and package")
                generated_code.append("data, desc = pyrsm.load_data('dataset_name', 'package_name')")

        # Pattern: Regression
        elif 'regression' in prompt_lower or 'regress' in prompt_lower:
            df_names = list(dataframes.keys())
            if df_names:
                df_name = df_names[0]  # Use first available
                df_info = dataframes[df_name]

                # Extract variables from prompt
                rvar = 'price' if 'price' in prompt_lower else (df_info.get('columns', ['y'])[0])
                evar = []
                if 'carat' in prompt_lower:
                    evar.append('carat')
                if 'depth' in prompt_lower:
                    evar.append('depth')
                if 'table' in prompt_lower:
                    evar.append('table')

                if not evar and df_info.get('columns'):
                    cols = df_info['columns']
                    evar = cols[1:min(4, len(cols))]

                comments.append(f"# Regression: {rvar} ~ {' + '.join(evar)}")
                generated_code.append(f"reg = pyrsm.model.regress({{'{df_name}': {df_name}}}, rvar='{rvar}', evar={evar})")
                generated_code.append("reg.summary()")
            else:
                comments.append("# No dataset loaded")
                generated_code.append("# Load a dataset first: data, _ = pyrsm.load_data('diamonds', 'model')")

        # Pattern: Summary statistics
        elif 'summary' in prompt_lower or 'describe' in prompt_lower:
            df_names = list(dataframes.keys())
            if df_names:
                df_name = df_names[0]
                comments.append(f"# Summary statistics for {df_name}")
                generated_code.append(f"{df_name}.describe()")
            else:
                generated_code.append("# Load data first")

        # Default
        else:
            comments.append(f"# Generated from: {prompt}")
            generated_code.append(f"# Available datasets: {', '.join(dataframes.keys())}")
            generated_code.append("# Try: 'load diamonds dataset', 'fit regression', 'summary statistics'")

        result = "\n".join(comments) + "\n" + "\n".join(generated_code) if comments else "\n".join(generated_code)

        result_text = f"Generated code from prompt: '{prompt}'\n\n```python\n{result}\n```\n"
        result_text += "\n**Suggested next steps:**\n"
        result_text += "- Review the generated code\n"
        result_text += "- Execute it in your notebook\n"
        result_text += "- Modify as needed for your analysis\n"

        return [TextContent(type="text", text=result_text)]

    if name == "single_mean":
        data_name = arguments["data_name"]
        var = arguments["var"]
        alt_hyp = arguments.get("alt_hyp", "two-sided")
        conf = arguments.get("conf", 0.95)
        comp_value = arguments.get("comp_value", 0)
        dec = arguments.get("dec", 3)
        plots = arguments.get("plots", [])

        # Get data from store
        if data_name not in DATA_STORE:
            return [TextContent(type="text", text=f"Error: Dataset '{data_name}' not loaded. Use data_load or data_load_file first.")]

        data = DATA_STORE[data_name]

        # Check if variable exists
        if var not in data.columns:
            return [TextContent(type="text", text=f"Error: Variable '{var}' not found in dataset '{data_name}'. Available: {list(data.columns)}")]

        # Generate code
        code_lines = [
            "import pyrsm",
            f"# Single mean test: {var}",
            f"sm = pyrsm.basics.single_mean({{'{data_name}': {data_name}}}, var='{var}', alt_hyp='{alt_hyp}', conf={conf}, comp_value={comp_value})",
            f"sm.summary(dec={dec})"
        ]

        if plots:
            code_lines.append(f"sm.plot(plots={plots})")

        generated_code = "\n".join(code_lines)

        # Execute and capture output
        buffer = StringIO()
        try:
            with redirect_stdout(buffer):
                exec_globals = {data_name: data}
                exec(generated_code, exec_globals)
            output = buffer.getvalue()
        except Exception as e:
            output = f"Error executing: {str(e)}"

        result = f"**Single Mean Hypothesis Test**\n\n"
        result += f"Generated code:\n```python\n{generated_code}\n```\n\n"
        result += f"Output:\n```\n{output}\n```\n\n"
        result += "**Suggested next steps:**\n"
        result += "- Interpret the p-value and confidence interval\n"
        result += "- Try different alternative hypotheses if needed\n"
        result += "- Plot the results with plots=['hist']\n"

        return [TextContent(type="text", text=result)]

    if name == "compare_means":
        data_name = arguments["data_name"]
        var1 = arguments["var1"]
        var2 = arguments["var2"]
        alt_hyp = arguments.get("alt_hyp", "two-sided")
        conf = arguments.get("conf", 0.95)
        sample_type = arguments.get("sample_type", "independent")
        test_type = arguments.get("test_type", "t-test")
        dec = arguments.get("dec", 3)
        plots = arguments.get("plots", [])

        # Get data from store
        if data_name not in DATA_STORE:
            return [TextContent(type="text", text=f"Error: Dataset '{data_name}' not loaded. Use data_load or data_load_file first.")]

        data = DATA_STORE[data_name]

        # Check if variables exist
        if var1 not in data.columns:
            return [TextContent(type="text", text=f"Error: Variable '{var1}' not found in dataset '{data_name}'. Available: {list(data.columns)}")]
        if var2 not in data.columns:
            return [TextContent(type="text", text=f"Error: Variable '{var2}' not found in dataset '{data_name}'. Available: {list(data.columns)}")]

        # Generate code
        code_lines = [
            "import pyrsm",
            f"# Compare means: {var1} vs {var2}",
            f"cm = pyrsm.basics.compare_means({{'{data_name}': {data_name}}}, var1='{var1}', var2='{var2}', alt_hyp='{alt_hyp}', conf={conf}, sample_type='{sample_type}', test_type='{test_type}')",
            f"cm.summary(dec={dec})"
        ]

        if plots:
            code_lines.append(f"cm.plot(plots={plots})")

        generated_code = "\n".join(code_lines)

        # Execute and capture output
        buffer = StringIO()
        try:
            with redirect_stdout(buffer):
                exec_globals = {data_name: data}
                exec(generated_code, exec_globals)
            output = buffer.getvalue()
        except Exception as e:
            output = f"Error executing: {str(e)}"

        result = f"**Compare Means Test**\n\n"
        result += f"Generated code:\n```python\n{generated_code}\n```\n\n"
        result += f"Output:\n```\n{output}\n```\n\n"
        result += "**Suggested next steps:**\n"
        result += "- Interpret the p-value and mean differences\n"
        result += "- Check if assumptions are met (normality, equal variance)\n"
        result += "- Try non-parametric test (test_type='wilcox') if needed\n"
        result += "- Visualize with plots=['box'] or plots=['density']\n"

        return [TextContent(type="text", text=result)]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
