# pyrsm MCP Server

MCP server exposing pyrsm statistical tools to AI assistants with **state management** for data and models.

## Features

✅ **Data Loading** - Load pyrsm's 50+ built-in datasets
✅ **Data Registry** - Loaded datasets persist across requests
✅ **Model Registry** - Fitted models stored for reuse
✅ **No Refitting** - Get diagnostics/plots without re-running
✅ **Code Generation** - Returns executable pyrsm code
✅ **Smart Suggestions** - Context-aware next step recommendations

## Available Tools

### Data Management
- `data_load` - Load datasets from pyrsm library
- `data_list` - List loaded datasets
- `data_info` - Get dataset details (columns, types, preview)

### Regression Analysis
- `regress_fit` - Fit model and store for reuse
- `regress_summary` - Get statistics (no refitting!)
- `regress_plot` - Generate plots (no refitting!)
- `regress_list` - List stored models

## Quick Start

```bash
cd /home/vnijs/gh/pyrsm/mcp-server

# Setup
make install

# Test data loading
source .venv/bin/activate
python test_data_loading.py

# Test regression with state
python test_regression.py
```

## Example Workflow

```
User: "What datasets are available in the model package?"
AI calls: data_load(package='model')
Returns: List of 9 datasets

User: "Load the diamonds dataset"
AI calls: data_load(name='diamonds', package='model')
Returns: Dataset loaded (3000 rows × 11 columns)

User: "Fit a regression: price explained by carat, depth, and table"
AI calls: regress_fit(data_name='diamonds', rvar='price', evar=['carat','depth','table'])
Returns: Model ID + summary

User: "Add VIF"
AI calls: regress_summary(model_id='reg_xyz', vif=True)
Returns: Updated summary (NO REFITTING!)

User: "Show diagnostic plots"
AI calls: regress_plot(model_id='reg_xyz', plot_type='dashboard')
Returns: Plot code + Suggested next steps (interpret patterns, check observations)
```

**Suggested Next Steps** appear after every tool response, guiding users through proper analysis workflow!

## Configuration

Currently configured to use: **Regression Server with Data Loading**

### VS Code
Config at: `~/.config/Code/User/mcp.json`

### Claude Code
Config at: `.mcp.json` in project root

**Restart your AI assistant** to load all tools.

## Architecture

### Dual Registry System

```
DATA_STORE = {
    'sample': <DataFrame>,
    'diamonds': <DataFrame>,
    ...
}

MODEL_STORE = {
    'reg_abc123': {
        'obj': <regression object>,
        'data_name': 'diamonds',
        'rvar': 'price',
        'evar': ['carat', 'depth', 'table']
    }
}
```

### Benefits

- **Load once, use many times** - Dataset persists for multiple analyses
- **Fit once, explore many times** - Model persists for diagnostics/plots
- **No unnecessary computation** - State management prevents redundant work
- **Natural workflow** - Mirrors how analysts actually work

## Files

- `server_regression.py` - Main MCP server with data + regression tools
- `test_data_loading.py` - Data workflow demonstration
- `test_regression.py` - Regression workflow demonstration
- `REGRESSION_TOOLS.md` - Detailed tool documentation
- `Makefile` - Quick commands

## Available Datasets

pyrsm includes 50+ datasets across categories:
- **basics**: demand_uk, salary, shopping, titanic...
- **model**: diamonds, houseprices, catalog, dvd...
- **data**: flights, zillow, newspaper...
- **design**: carpet, mba...

Use `data_load()` without name to list all available.

## Next Steps

1. Test in Claude Code: "What datasets are available?"
2. Load data: "Load the diamonds dataset"
3. Analyze: "Fit a regression with price as response"
4. Explore: "Add VIF" → "Show diagnostic plots"
