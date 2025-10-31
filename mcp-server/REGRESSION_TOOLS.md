# Regression Tools - State Management Implementation

## Overview

The regression MCP server wraps `pyrsm.model.regress` with state management, allowing students to:
1. Fit a model once
2. Request additional statistics/plots without refitting
3. Work with multiple models simultaneously

## Architecture

### Model Registry
```python
MODEL_STORE = {
    'reg_abc123_1234567': {
        'obj': <pyrsm regress object>,
        'rvar': 'sales',
        'evar': ['x1', 'x2', 'x3'],
        'fitted_at': '2025-10-27 02:15:30',
        'metadata': {}
    }
}
```

### Tool Flow

```
┌─────────────────┐
│  regress_fit    │  Fits model, stores in registry
│   rvar, evar    │  Returns: model_id + summary
└────────┬────────┘
         │
         ├─────> MODEL_STORE[model_id] = {obj, metadata}
         │
         v
┌─────────────────┐
│ regress_summary │  Retrieves stored model
│   model_id      │  Calls: obj.summary(vif=True)
└────────┬────────┘  NO REFITTING!
         │
         v
┌─────────────────┐
│  regress_plot   │  Retrieves stored model
│   model_id      │  Calls: obj.plot(plots='dashboard')
└─────────────────┘  NO REFITTING!
```

## Available Tools

### 1. `regress_fit`

**Purpose:** Fit regression model and store for reuse

**Parameters:**
- `rvar` (required): Response variable name
- `evar` (required): List of explanatory variables
- `show_summary` (optional): Show summary immediately (default: true)
- `vif` (optional): Include VIF in initial summary (default: false)

**Returns:**
- Model ID for later reference
- Generated pyrsm code
- Summary output (if requested)

**Example:**
```
User: "Fit regression: sales explained by x1, x2, x3"

Tool Call:
regress_fit(rvar='sales', evar=['x1','x2','x3'], show_summary=True)

Response:
✓ Model fitted and stored as: reg_a1b2c3_1698765432

Response: sales
Predictors: x1, x2, x3

Generated code:
```python
reg = pyrsm.model.regress(data, rvar='sales', evar=['x1', 'x2', 'x3'])
```

Summary:
[Full regression output]

💡 Use model_id 'reg_a1b2c3_1698765432' for:
  • regress_summary - Get additional statistics
  • regress_plot - Generate diagnostic plots
```

### 2. `regress_summary`

**Purpose:** Get statistics from stored model (NO refitting)

**Parameters:**
- `model_id` (required): ID from regress_fit
- `vif` (optional): Show VIF (default: false)
- `fit` (optional): Show fit statistics (default: true)
- `dec` (optional): Decimal places (default: 3)

**Example:**
```
User: "Add VIF to check multicollinearity"

Tool Call:
regress_summary(model_id='reg_a1b2c3_1698765432', vif=True)

Response:
Summary for model: reg_a1b2c3_1698765432
(fitted at 2025-10-27 02:15:30)

Code:
```python
reg.summary(vif=True, fit=True, dec=3)
```

Output:
[Summary with VIF]
```

### 3. `regress_plot`

**Purpose:** Generate diagnostic plots (NO refitting)

**Parameters:**
- `model_id` (required): ID from regress_fit
- `plot_type` (required): 'dashboard', 'vimp', or 'pred'

**Plot Types:**
- `dashboard` - Residual diagnostics (4 plots)
- `vimp` - Variable importance
- `pred` - Prediction plot

**Example:**
```
User: "Show residual diagnostics"

Tool Call:
regress_plot(model_id='reg_a1b2c3_1698765432', plot_type='dashboard')

Response:
Plot type 'dashboard' for model: reg_a1b2c3_1698765432
(fitted at 2025-10-27 02:15:30)

Code:
```python
reg.plot(plots='dashboard')
```

📊 [Plot would be rendered here]
```

### 4. `regress_list`

**Purpose:** List all stored models

**Example:**
```
Tool Call:
regress_list()

Response:
Stored models (2):

• reg_a1b2c3_1698765432
  Response: sales
  Predictors: x1, x2, x3
  Fitted: 2025-10-27 02:15:30

• reg_d4e5f6_1698765450
  Response: price
  Predictors: x1, x2
  Fitted: 2025-10-27 02:17:10
```

## User Workflows

### Novice: Guided Iteration
```
1. "Run a regression on sales"
   → AI asks for variables
   → Fits with defaults
   → Shows basic summary

2. "What's VIF?"
   → AI explains
   → Offers to add VIF

3. "Yes, show VIF"
   → Calls regress_summary(vif=True)
   → NO refitting!

4. "Show me diagnostic plots"
   → Calls regress_plot(plot_type='dashboard')
   → NO refitting!
```

### Expert: Single Comprehensive Request
```
"Regress sales on x1-x3, show summary with VIF and residual plots"

AI interprets:
1. regress_fit(rvar='sales', evar=['x1','x2','x3'], vif=True)
2. regress_plot(model_id=..., plot_type='dashboard')

Returns everything at once
```

### Mixed: Exploratory Analysis
```
1. Fit model A: sales ~ x1 + x2
2. Fit model B: sales ~ x1 + x2 + x3
3. Compare: regress_summary for both
4. Best model? Add diagnostics
```

## Implementation Details

### Model ID Generation
```python
def generate_model_id(rvar: str, evar: list, timestamp: float) -> str:
    vars_str = f"{rvar}_{'_'.join(sorted(evar))}"
    hash_obj = hashlib.md5(vars_str.encode())
    return f"reg_{hash_obj.hexdigest()[:8]}_{int(timestamp)}"
```

### Storage
```python
def store_model(model_obj, rvar: str, evar: list, **kwargs) -> str:
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
```

### Retrieval
```python
def get_model(model_id: str):
    if model_id not in MODEL_STORE:
        raise ValueError(f"Model '{model_id}' not found")
    return MODEL_STORE[model_id]
```

## Benefits

✅ **Efficiency** - No unnecessary refitting
✅ **Exploration** - Easy to try different diagnostics
✅ **Learning** - Students see the iterative analysis process
✅ **Code Generation** - Always returns executable pyrsm code
✅ **Multi-model** - Compare different specifications

## Limitations (Current)

⚠️ **Memory only** - Models lost when server restarts
⚠️ **No persistence** - Can't save/load across sessions (yet)
⚠️ **Fixed data** - Uses sample data (real data loading coming)
⚠️ **No plots** - Plot rendering not implemented (returns code)

## Next Phase

1. Add data loading tools
2. Implement plot rendering (base64 encoding)
3. Add model persistence (pickle to disk)
4. Interactive form generation for variable selection
5. Add more model types (logistic, rforest, etc.)
