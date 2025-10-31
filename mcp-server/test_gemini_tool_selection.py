#!/usr/bin/env python3
"""
Checkpoint 3: Test Gemini LLM selecting correct MCP tools
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

import google.generativeai as genai

print("="*70)
print("CHECKPOINT 3: Gemini LLM Tool Selection")
print("="*70)

# Load API key
env_file = Path.home() / '.env'
for line in env_file.read_text().splitlines():
    if '=' in line and not line.strip().startswith('#'):
        key, value = line.split('=', 1)
        os.environ[key.strip()] = value.strip()

api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

# Convert MCP tools to Gemini format
tools = [
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
                            "description": "Name of the loaded dataset to use for hypothesis testing",
                        },
                        "var": {
                            "type": "string",
                            "description": "The variable/column name to test (must be numeric)",
                        },
                        "comp_value": {
                            "type": "number",
                            "description": "The comparison value for the test (value under null hypothesis)",
                        },
                        "alt_hyp": {
                            "type": "string",
                            "description": "The alternative hypothesis: two-sided, greater, or less",
                        },
                        "conf": {
                            "type": "number",
                            "description": "The confidence level for the test (e.g., 0.95 for 95%)",
                        }
                    },
                    "required": ["data_name", "var"]
                }
            },
            {
                "name": "compare_means",
                "description": "Compare means between groups using t-test or Wilcoxon test to determine if there are significant differences",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_name": {
                            "type": "string",
                            "description": "Name of the loaded dataset to use",
                        },
                        "var1": {
                            "type": "string",
                            "description": "First variable (can be categorical for grouping)",
                        },
                        "var2": {
                            "type": "string",
                            "description": "Second variable (must be numeric - this is what we're comparing)",
                        },
                        "test_type": {
                            "type": "string",
                            "description": "Type of test: t-test or wilcox",
                        }
                    },
                    "required": ["data_name", "var1", "var2"]
                }
            },
            {
                "name": "regress_fit",
                "description": "Fit a linear regression model to understand relationships between variables",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_name": {
                            "type": "string",
                            "description": "Name of loaded dataset to use",
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
                        "vif": {
                            "type": "boolean",
                            "description": "Include VIF to check multicollinearity",
                        }
                    },
                    "required": ["data_name", "rvar", "evar"]
                }
            }
        ]
    }
]

# Create model with tools
model = genai.GenerativeModel('gemini-2.0-flash-exp', tools=tools)

# Test cases
test_cases = [
    {
        "prompt": "I have a salary dataset loaded. Test if the mean salary equals 100000.",
        "expected_tool": "single_mean",
        "context": "Available data: salary dataset with columns: salary, rank, discipline"
    },
    {
        "prompt": "Compare the salary between different academic ranks (Professor vs Assistant Professor).",
        "expected_tool": "compare_means",
        "context": "Available data: salary dataset with columns: salary (numeric), rank (categorical)"
    },
    {
        "prompt": "Fit a regression model to predict price using carat, depth, and table as predictors.",
        "expected_tool": "regress_fit",
        "context": "Available data: diamonds dataset with columns: price, carat, depth, table, cut, color"
    },
    {
        "prompt": "Run a hypothesis test to see if average age is 30.",
        "expected_tool": "single_mean",
        "context": "Available data: people dataset with age column"
    },
    {
        "prompt": "Check if there's a difference in sales between product A and product B.",
        "expected_tool": "compare_means",
        "context": "Available data: sales dataset with product (categorical) and sales (numeric) columns"
    }
]

print("\n" + "="*70)
print("TESTING LLM TOOL SELECTION")
print("="*70)

results = []
for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. TEST CASE")
    print("-"*70)
    print(f"Prompt: {test['prompt']}")
    print(f"Context: {test['context']}")
    print(f"Expected tool: {test['expected_tool']}")

    # Create full prompt with context
    full_prompt = f"{test['context']}\n\nUser request: {test['prompt']}"

    try:
        response = model.generate_content(full_prompt)

        # Check if tool was called
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call'):
                    tool_name = part.function_call.name
                    args = dict(part.function_call.args)

                    correct = tool_name == test['expected_tool']
                    status = "✓" if correct else "✗"

                    print(f"{status} LLM selected: {tool_name}")
                    print(f"  Arguments: {args}")

                    results.append({
                        'test': i,
                        'expected': test['expected_tool'],
                        'actual': tool_name,
                        'correct': correct
                    })
                    break
        else:
            print("✗ No tool call generated")
            print(f"  Response: {response.text}")
            results.append({
                'test': i,
                'expected': test['expected_tool'],
                'actual': 'none',
                'correct': False
            })

    except Exception as e:
        print(f"✗ Error: {e}")
        results.append({
            'test': i,
            'expected': test['expected_tool'],
            'actual': 'error',
            'correct': False
        })

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

correct = sum(1 for r in results if r['correct'])
total = len(results)
accuracy = (correct / total * 100) if total > 0 else 0

print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
print()

for r in results:
    status = "✓" if r['correct'] else "✗"
    print(f"{status} Test {r['test']}: Expected {r['expected']}, Got {r['actual']}")

if accuracy >= 80:
    print("\n" + "="*70)
    print("✓ CHECKPOINT 3 COMPLETE: LLM tool selection working!")
    print("="*70)
    print("\nGemini successfully selects correct tools from natural language.")
    print("Ready for Checkpoint 4: Build %%mcp magic bridge")
else:
    print("\n" + "="*70)
    print("⚠ CHECKPOINT 3 NEEDS IMPROVEMENT")
    print("="*70)
    print(f"LLM accuracy: {accuracy:.1f}% (target: 80%+)")
    print("May need to refine tool descriptions or prompts")
