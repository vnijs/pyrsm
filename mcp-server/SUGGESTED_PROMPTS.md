# Suggested Next Steps Feature

## Overview

Every MCP tool response now includes **context-aware suggested next steps** to guide users through their analysis workflow.

## How It Works

After each tool call, the server returns:
1. Tool output (data, summary, plots, etc.)
2. Generated code
3. **Contextual suggestions** for what to do next

## Examples

### After Loading Data

```
✓ Dataset loaded: diamonds

**Suggested next steps:**
- Explore the data structure and variables
- Run a regression analysis on this dataset
- Ask me to explain what this dataset contains
```

### After Fitting Model

```
✓ Model fitted and stored as: reg_abc123

**Suggested next steps:**
- Check for multicollinearity (VIF)
- View residual diagnostics (dashboard plot)
- Check variable importance (permutation importance)
- Interpret the coefficients and p-values
```

### After Getting Summary

```
Summary for model: reg_abc123
[...output...]

**Suggested next steps:**
- Check for multicollinearity (VIF)  [only if not already shown]
- View residual diagnostics (dashboard plot)
- Check variable importance (permutation importance)
- Interpret the statistical significance of coefficients
```

### After Dashboard Plot

```
Plot type 'dashboard' for model: reg_abc123
[...plot code...]

**Suggested next steps:**
- Interpret residual patterns (look for non-linearity, heteroscedasticity)
- Check for influential observations
- Try variable importance plot if model fits well
```

### After Variable Importance Plot

```
Plot type 'vimp' for model: reg_abc123
[...plot code...]

**Suggested next steps:**
- Identify which variables matter most
- Consider simplifying model by removing unimportant variables
- Check residual diagnostics if you haven't already
```

## Benefits

### For Novices
- **Guided learning** - Suggestions teach proper analysis workflow
- **Reduced cognitive load** - Don't need to remember all options
- **Natural progression** - Each step suggests logical next steps
- **Discovery** - Learn about features they didn't know existed

### For Experts
- **Quick reminders** - Helpful prompts for complete analysis
- **Efficiency** - Can quickly click/select instead of typing
- **Quality assurance** - Ensures no diagnostic steps are missed

### For Teaching
- **Best practices** - Embeds good statistical workflow
- **Exploration** - Encourages checking assumptions
- **Interpretation** - Reminds users to interpret, not just compute

## Adaptive Suggestions

Suggestions are **context-aware**:

- **VIF already shown?** → Don't suggest it again
- **Just viewed dashboard?** → Suggest interpretation next
- **Just viewed vimp?** → Suggest model simplification

## User Experience Flow

```
1. User: "Load diamonds dataset"
   → Suggests: Explore, analyze, or learn about it

2. User: "Fit regression: price ~ carat + depth"
   → Suggests: Check VIF, residuals, importance

3. User: "Check VIF"
   → Suggests: Residuals, importance, interpretation
   (VIF not suggested again since already done)

4. User: "Show dashboard plot"
   → Suggests: Interpret patterns, check observations

5. User: "Variable importance plot"
   → Suggests: Identify key vars, simplify model
```

## Implementation

Each tool response includes:

```python
result += "**Suggested next steps:**\n"
result += f"- Action 1 with specific guidance\n"
result += f"- Action 2 with context\n"
result += f"- Action 3\n"
```

AI assistants can render these as:
- **Claude Code**: Clickable markdown bullets
- **VS Code**: Interactive quick picks
- **Shiny**: Clickable suggestion chips
- **Plain text**: Simple bulleted list

## Future Enhancements

- **Personalization**: Remember user's skill level, adjust suggestions
- **Smart ordering**: Prioritize based on what's most important for this specific model
- **Conditional logic**: More sophisticated context awareness
- **Interactive buttons**: Rich UI elements in supported clients
- **Tutorial mode**: More detailed explanations for novices
