# Interactive UI Features

## Overview

The MCP bridge magic now includes enhanced output formatting and interactive UI elements for better user experience.

## Features

### 1. Markdown Text Wrapping

**Problem**: LLM text responses would run off the screen without wrapping.

**Solution**: All text output now uses `IPython.display.Markdown()` for proper formatting and wrapping.

**Example**:
```python
%%mcp
What is this dataset about?
```

Text responses now wrap properly and support markdown formatting.

### 2. Interactive Dataset Selection

**Problem**: When LLM asks "which dataset?", users had to manually type `%mcp_use <dataset>`.

**Solution**: Automatic dropdown widget appears with "Set Active & Retry" button.

**Triggers when LLM response contains**:
- "which dataset"
- "select dataset"
- "what dataset"
- "choose dataset"

**Behavior**:
1. Shows dropdown with available datasets
2. Click "Set Active & Retry" button
3. Automatically sets active dataset and re-runs your original prompt

**Example**:
```python
%%mcp
Test if mean equals 100000
```

If no active dataset set, you'll see:
```
Which dataset would you like to use?

[Dropdown: salary ▼] [Set Active & Retry]
```

### 3. Clickable Next Step Buttons

**Problem**: "Suggested next steps" were just text - users had to type them manually.

**Solution**: Each suggestion becomes a clickable button that auto-executes the step.

**Features**:
- Buttons appear below the analysis output
- Click any button to automatically run that analysis
- Tooltips show the actual `%%mcp` prompt that will be executed
- Context-aware: buttons know which model to use

**Example Output**:
```markdown
**Suggested next steps:**
- Check for multicollinearity (VIF)
- View residual diagnostics (dashboard plot)
- Check variable importance (permutation importance)

**Quick actions:**
[Check for multicollinearity (VIF)] [View residual diagnostics (dashboard plot)] [Check variable importance (permutation importance)]
```

Click any button → automatically executes that analysis!

## Smart Context Awareness

The button system is intelligent:

### Model Detection
```python
%%mcp
Predict price from carat using diamonds

# After this, 'reg' model exists
# Next step buttons will automatically include 'reg' in prompts:
# "Calculate VIF scores for reg"
# "Show dashboard plot for reg"
```

### Step Text Parsing
The system converts natural language suggestions to proper prompts:

| Suggestion Text | Generated Prompt |
|----------------|------------------|
| "Check for multicollinearity (VIF)" | "Calculate VIF scores for reg" |
| "View residual diagnostics (dashboard plot)" | "Show residual dashboard for reg" |
| "Check variable importance (permutation importance)" | "Show variable importance plot for reg" |

## Implementation Details

### Functions Added

1. **`_display_markdown(text)`**
   - Wrapper for `display(Markdown(text))`
   - Ensures text wrapping

2. **`_show_dataset_selector(original_prompt)`**
   - Creates interactive dropdown + button
   - Stores original prompt in `LAST_PROMPT`
   - Re-executes prompt after dataset selection

3. **`_enhance_next_steps(result_text, context)`**
   - Parses "Suggested next steps" section
   - Creates buttons with click handlers
   - Returns markdown + button list

4. **`_step_to_prompt(step_text, context)`**
   - Converts suggestion text to %%mcp prompt
   - Uses context to find active model
   - Pattern matches common operations

### Global Variables

- **`ACTIVE_DATASET`**: Current active dataset name
- **`LAST_PROMPT`**: Stores prompt for re-execution after dataset selection

## Usage Tips

### For Students

1. **Let the UI guide you**: Click suggested next step buttons to explore analysis workflows

2. **No typing needed**: Dataset selector and action buttons handle everything

3. **Learn by doing**: Button tooltips show what prompt they'll execute

### For Instructors

1. **Guided learning**: Next step suggestions create structured learning paths

2. **Reduce friction**: Students don't need to remember exact command syntax

3. **Progressive complexity**: Suggestions build on previous analysis steps

## Testing

Test the interactive features:

```python
# 1. Test dataset selector
%%mcp
Test if mean equals 100

# Should show dropdown if no active dataset

# 2. Test next step buttons
%mcp_use salary
%%mcp
Compare salary between ranks

# Click the suggested next step buttons

# 3. Test button auto-execution
%mcp_use diamonds
%%mcp
Predict price from carat

# Should see buttons for VIF, residuals, variable importance
# Click any button - it should auto-execute
```

## Browser Compatibility

Works in:
- Jupyter Notebook (classic)
- JupyterLab
- VS Code with Jupyter extension
- Google Colab

Requires `ipywidgets` installed (already in pyrsm dependencies).
