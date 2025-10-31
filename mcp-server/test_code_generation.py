#!/usr/bin/env python3
"""
Test the code generation logic (without IPython)
"""

print("="*70)
print("CODE GENERATION LOGIC TEST")
print("="*70)

# Inline the generation function for testing
def generate_code_from_prompt(prompt, context):
    """Generate pyrsm code from natural language prompt."""
    prompt_lower = prompt.lower().strip()
    generated_code = []
    comments = []

    dataframes = context.get('dataframes', {})

    # Pattern: Load data
    if 'load' in prompt_lower and 'data' in prompt_lower:
        if 'diamonds' in prompt_lower:
            comments.append("# Load diamonds dataset")
            generated_code.append("diamonds, desc = pyrsm.load_data('diamonds', 'model')")
            generated_code.append("print(f'Loaded: {diamonds.shape[0]} rows × {diamonds.shape[1]} columns')")
        else:
            comments.append("# Load dataset")
            generated_code.append("data, desc = pyrsm.load_data('dataset_name', 'package_name')")

    # Pattern: Regression
    elif 'regression' in prompt_lower or 'regress' in prompt_lower:
        df_names = list(dataframes.keys())
        if df_names:
            df_name = df_names[0]
            df_info = dataframes[df_name]

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
            generated_code.append(f"reg = pyrsm.model.regress({df_name}, rvar='{rvar}', evar={evar})")
            generated_code.append("reg.summary()")
        else:
            comments.append("# No dataset loaded")
            generated_code.append("# Load a dataset first")

    # Pattern: Summary
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

    result = "\n".join(comments) + "\n" + "\n".join(generated_code) if comments else "\n".join(generated_code)
    return result


# Test cases
test_cases = [
    {
        'prompt': 'Load the diamonds dataset',
        'context': {},
        'expected': 'diamonds'
    },
    {
        'prompt': 'Fit a regression with price as response and carat, depth as predictors',
        'context': {
            'dataframes': {
                'diamonds': {
                    'shape': (3000, 11),
                    'columns': ['price', 'carat', 'depth', 'table', 'x', 'y', 'z']
                }
            }
        },
        'expected': 'regress'
    },
    {
        'prompt': 'Show summary statistics',
        'context': {
            'dataframes': {
                'diamonds': {
                    'shape': (3000, 11),
                    'columns': ['price', 'carat']
                }
            }
        },
        'expected': 'describe'
    },
]

print()
for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['prompt']}")
    print("-" * 70)
    code = generate_code_from_prompt(test['prompt'], test['context'])
    print(code)

    if test['expected'] in code:
        print(f"✓ Contains '{test['expected']}'")
    else:
        print(f"✗ Missing '{test['expected']}'")
    print()

print("="*70)
print("✓ CODE GENERATION TESTS COMPLETE")
print("="*70)
print()
print("Next step: Open the notebook in VS Code!")
print("  File: /home/vnijs/gh/pyrsm/mcp-server/examples/prompt_demo.ipynb")
