# Quick Start: Prompt Magic in VS Code

## Step-by-Step Instructions

### 1. Open the Demo Notebook in VS Code

```bash
code /home/vnijs/gh/pyrsm/mcp-server/examples/prompt_demo.ipynb
```

### 2. Select Python Kernel

- Click "Select Kernel" in top right
- Choose: `/home/vnijs/gh/pyrsm/mcp-server/.venv/bin/python`

### 3. Run Cell 1: Load Extension

```python
import sys
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')
%load_ext prompt_magic
```

**Expected output**: `✓ Prompt magic loaded!`

### 4. Run Cell 2: First Prompt

```python
%%prompt
Load the diamonds dataset
```

**What happens**:
1. New cell appears below with generated code
2. Code executes automatically
3. You see: `Loaded: 3000 rows × 11 columns`

### 5. Run Cell 3: Fit Regression

```python
%%prompt
Fit a regression with price as response and carat, depth, table as predictors
```

**What happens**:
1. Generated code: `reg = pyrsm.model.regress(diamonds, rvar='price', evar=['carat', 'depth', 'table'])`
2. Auto-executes
3. Shows regression summary output

### 6. Experiment!

Try your own prompts:

```python
%%prompt
Show summary statistics for the diamonds dataset
```

```python
%%prompt
Fit a regression with just carat as predictor
```

## Troubleshooting

**Issue**: "No module named 'prompt_magic'"
- **Fix**: Make sure you ran the `sys.path.insert()` line first

**Issue**: "NameError: name 'diamonds' is not defined"
- **Fix**: Run the "Load the diamonds dataset" prompt first

**Issue**: Generated code doesn't appear
- **Fix**: The code appears in a NEW cell below. Scroll down!

**Issue**: Code doesn't auto-execute
- **Fix**: Check if VS Code blocked execution. Click "Run Cell" manually if needed.

## What's Working

✅ Pattern recognition for:
- Loading datasets (diamonds, salary, titanic, etc.)
- Fitting regression models
- Summary statistics
- Variable extraction from prompts

✅ Context awareness:
- Sees loaded DataFrames
- Knows what columns exist
- Uses first available data if not specified

✅ Auto-execution:
- Generated code runs immediately
- You can still edit it afterwards

## Files Created

- `prompt_magic.py` - Main extension
- `examples/prompt_demo.ipynb` - Demo notebook
- `PROMPT_MAGIC_README.md` - Full documentation
- `test_code_generation.py` - Tests (all passing ✓)
