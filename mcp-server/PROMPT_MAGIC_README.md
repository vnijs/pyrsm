# Prompt Magic - Natural Language Code Generation for pyrsm

Generate and execute pyrsm analysis code from natural language prompts in Jupyter notebooks.

## Quick Start

### 1. Load the Extension

```python
import sys
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')
%load_ext prompt_magic
```

### 2. Write a Prompt

```python
%%prompt
Load the diamonds dataset
```

**Result**: Code is generated, inserted below, and executed automatically:
```python
# Load diamonds dataset
diamonds, desc = pyrsm.load_data('diamonds', 'model')
print(f'Loaded: {diamonds.shape[0]} rows × {diamonds.shape[1]} columns')
```

### 3. Continue Your Analysis

```python
%%prompt
Fit a regression with price as response and carat, depth as predictors
```

**Result**:
```python
# Regression: price ~ carat + depth
reg = pyrsm.model.regress(diamonds, rvar='price', evar=['carat', 'depth'])
reg.summary()
```

## How It Works

1. **Context Awareness**: The magic inspects your notebook's namespace (loaded DataFrames, models, imports)
2. **Pattern Matching**: Recognizes common analysis patterns (load data, fit regression, summary stats)
3. **Code Generation**: Creates executable pyrsm code
4. **Auto-Execution**: Inserts code in new cell and runs it immediately

## Supported Prompts (Prototype v1)

| Prompt Pattern | Example | Generated Code |
|----------------|---------|----------------|
| Load dataset | `Load the diamonds dataset` | `pyrsm.load_data('diamonds', 'model')` |
| Fit regression | `Fit regression with price and carat, depth` | `pyrsm.model.regress(data, rvar='price', evar=['carat', 'depth'])` |
| Summary stats | `Show summary statistics` | `dataframe.describe()` |
| Compare means | `Compare means between groups` | `pyrsm.basics.compare_means(...)` |

## Example Workflow

```python
# Load extension
%load_ext prompt_magic

# Prompt 1: Load data
%%prompt
Load the diamonds dataset
# → Auto-generates and runs: diamonds, desc = pyrsm.load_data('diamonds', 'model')

# Prompt 2: Analyze
%%prompt
Fit a regression with price as response and carat, depth, table as predictors
# → Auto-generates and runs: reg = pyrsm.model.regress(diamonds, ...)

# Prompt 3: More analysis
%%prompt
Show summary statistics
# → Auto-generates and runs: diamonds.describe()
```

## Files

- **prompt_magic.py** - IPython extension (main implementation)
- **server_regression.py** - MCP server with `code_generate` tool
- **examples/prompt_demo.ipynb** - Demo notebook

## Installation

### Option 1: Manual Load (Quick Test)
```python
import sys
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')
%load_ext prompt_magic
```

### Option 2: Auto-Load (Permanent)
Add to `~/.ipython/profile_default/ipython_config.py`:
```python
c.InteractiveShellApp.extensions = ['prompt_magic']
```

And add to `sys.path` or install as package.

## Future Enhancements

- [ ] LLM integration for smarter code generation
- [ ] More pattern recognition (plots, hypothesis tests, etc.)
- [ ] Error handling with retry and suggestions
- [ ] Integration with MCP server for state management
- [ ] File loading from Django browser
- [ ] Quarto document support
- [ ] Interactive code preview before execution

## Testing

Open `examples/prompt_demo.ipynb` in VS Code and run through the cells.

## Requirements

- Python 3.10+
- IPython
- pandas
- pyrsm
- Jupyter/VS Code with Jupyter extension

## Compatibility

- ✅ VS Code with Jupyter extension
- ✅ JupyterLab
- ✅ Jupyter Notebook
- ⚠️ Google Colab (with manual load)
- ❌ Quarto (future support planned)
