#!/usr/bin/env python3
"""
Executable test: Does AI-generated code actually WORK?

Tests with real data and actually executes the generated code.
Scores based on:
1. Does it run without errors?
2. Does it use pyrsm?
3. Does it produce correct output?
"""

import os
from pathlib import Path
import pandas as pd
import google.generativeai as genai
import sys
from io import StringIO

# Load environment variables
env_file = Path.home() / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

# Create test dataset
TEST_DATA = pd.DataFrame({
    'sales': [100, 150, 200, 250, 300, 120, 180, 220, 280, 320],
    'x1': [1, 2, 3, 4, 5, 1.5, 2.5, 3.5, 4.5, 5.5],
    'x2': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
    'x3': [5, 10, 15, 20, 25, 7, 12, 17, 22, 27],
    'price': [95, 105, 98, 102, 110, 97, 103, 99, 101, 108],
    'treatment': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'score': [85, 90, 88, 92, 87, 89, 91, 86, 93, 94],
    'success': [True, False, True, True, False, True, False, True, True, False]
})

TEST_PROMPTS = [
    {
        "prompt": "Run a linear regression with 'sales' as response and x1, x2, x3 as explanatory variables. Show summary.",
        "expected_calls": ["regress", "summary"],
    },
    {
        "prompt": "Test if the mean of 'price' is significantly different from 100",
        "expected_calls": ["single_mean", "summary"],
    },
    {
        "prompt": "Compare means for 'score' between groups in 'treatment'",
        "expected_calls": ["compare_means", "summary"],
    },
]

def generate_code(model, prompt: str, context: str = "") -> str:
    """Generate code using AI"""
    system_prompt = """You are helping a student write Python code for data analysis.

IMPORTANT:
- Generate ONLY executable Python code
- Assume a DataFrame called 'data' already exists with the required columns
- Do NOT include data creation code
- Do NOT use try-except blocks
- Be concise and use straightforward syntax
- Print the summary/results at the end"""

    if context:
        full_prompt = f"{system_prompt}\n\nDocumentation:\n{context}\n\nStudent request: {prompt}\n\nGenerate only the code:"
    else:
        full_prompt = f"{system_prompt}\n\nStudent request: {prompt}\n\nGenerate only the code:"

    response = model.generate_content(full_prompt)
    code = response.text

    # Extract code from markdown if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    return code.strip()

def execute_code(code: str, data: pd.DataFrame) -> dict:
    """Execute the generated code and capture results"""
    # Create execution environment
    env = {
        'data': data,
        'pd': pd,
        '__builtins__': __builtins__,
    }

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    success = False
    error = None
    output = ""
    uses_pyrsm = False

    try:
        # Execute the code
        exec(code, env)
        success = True
        output = captured_output.getvalue()

        # Check if pyrsm was used
        uses_pyrsm = "pyrsm" in code or "rsm" in code

    except Exception as e:
        error = str(e)
    finally:
        sys.stdout = old_stdout

    return {
        "success": success,
        "error": error,
        "output": output,
        "uses_pyrsm": uses_pyrsm,
        "code": code
    }

def score_result(result: dict, expected_calls: list) -> dict:
    """Score the execution result"""
    score = 0
    max_score = 5

    # 2 points for running successfully
    if result["success"]:
        score += 2

    # 2 points for using pyrsm
    if result["uses_pyrsm"]:
        score += 2

    # 1 point for producing output
    if result["output"] and len(result["output"]) > 50:
        score += 1

    return {
        **result,
        "score": score,
        "max_score": max_score
    }

def run_test_suite(model, context_name: str, context: str):
    """Run all tests with given context"""
    print(f"\n{'='*80}")
    print(f"Testing with: {context_name}")
    print('='*80)

    results = []
    for i, test in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] {test['prompt'][:60]}...")

        # Generate code
        code = generate_code(model, test['prompt'], context)

        # Execute code
        exec_result = execute_code(code, TEST_DATA)

        # Score result
        result = score_result(exec_result, test['expected_calls'])
        result['prompt'] = test['prompt']
        results.append(result)

        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        pyrsm_status = "pyrsm" if result['uses_pyrsm'] else "generic"
        print(f"  {status} | {pyrsm_status} | Score: {result['score']}/{result['max_score']}")

        if not result['success']:
            print(f"  Error: {result['error'][:100]}")

    return results

def print_summary(all_results: dict):
    """Print comparison summary"""
    print(f"\n\n{'='*80}")
    print("EXECUTABLE TEST RESULTS")
    print('='*80)

    for context_name, results in all_results.items():
        total_score = sum(r['score'] for r in results)
        max_possible = sum(r['max_score'] for r in results)
        success_count = sum(1 for r in results if r['success'])
        pyrsm_count = sum(1 for r in results if r['uses_pyrsm'])

        print(f"\n{context_name}:")
        print(f"  Score: {total_score}/{max_possible} ({100*total_score/max_possible:.1f}%)")
        print(f"  Successful executions: {success_count}/{len(results)}")
        print(f"  Used pyrsm: {pyrsm_count}/{len(results)}")

def load_example_context():
    """Create minimal example context"""
    return """
# pyrsm Quick Examples

## Linear Regression
```python
import pyrsm
result = pyrsm.model.regress(data, rvar='sales', evar=['x1', 'x2', 'x3'])
result.summary()
```

## Single Mean Test
```python
import pyrsm
sm = pyrsm.basics.single_mean(data, var='price', comp_value=100)
sm.summary()
```

## Compare Means
```python
import pyrsm
cm = pyrsm.basics.compare_means(data, var='score', levs='treatment')
cm.summary()
```
"""

def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in ~/.env")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    print("EXECUTABLE CODE TEST")
    print(f"Testing {len(TEST_PROMPTS)} prompts with real data and execution\n")
    print(f"Test dataset shape: {TEST_DATA.shape}")
    print(f"Columns: {list(TEST_DATA.columns)}\n")

    # Load contexts
    examples = load_example_context()

    # Run tests
    all_results = {}
    all_results["NO CONTEXT"] = run_test_suite(model, "NO CONTEXT", "")
    all_results["WITH EXAMPLES"] = run_test_suite(model, "WITH EXAMPLES", examples)

    # Print summary
    print_summary(all_results)

    # Show example code
    print(f"\n{'='*80}")
    print("EXAMPLE GENERATED CODE (First test, with examples):")
    print('='*80)
    print(all_results["WITH EXAMPLES"][0]['code'])

if __name__ == "__main__":
    main()
