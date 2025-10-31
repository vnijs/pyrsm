#!/usr/bin/env python3
"""
Automated test: Does documentation context improve AI code generation for pyrsm?

Compares AI-generated code with:
1. NO context (baseline)
2. Current README context
3. Enhanced examples context

Uses APIs from ~/.env
"""

import os
from pathlib import Path
import google.generativeai as genai

# Load environment variables from ~/.env
env_file = Path.home() / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

# Test prompts
TEST_PROMPTS = [
    "I want to run a linear regression with 'sales' as the response and x1, x2, x3 as the explanatory variables. Show me the summary output",
    "Test if the mean of 'price' is significantly different from 100",
    "Compare means between two groups 'treatment' and 'control' for variable 'score'",
    "Load the diamonds dataset from pyrsm",
    "Test if proportion is different from 0.5",
]

def load_readme():
    """Load current README"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text()
    return ""

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

## Load Data
```python
import pyrsm
diamonds = pyrsm.load_data('diamonds')
```

## Single Proportion
```python
import pyrsm
sp = pyrsm.basics.single_prop(data, var='success', lev='yes', comp_value=0.5)
sp.summary()
```
"""

def score_response(response: str) -> dict:
    """Score the response"""
    response_lower = response.lower()

    uses_pyrsm = "pyrsm" in response_lower
    uses_import = "import pyrsm" in response_lower or "from pyrsm" in response_lower
    uses_correct_module = any(x in response_lower for x in ["pyrsm.model.", "pyrsm.basics."])
    has_method_call = ".summary()" in response_lower or ".plot()" in response_lower

    score = 0
    if uses_pyrsm:
        score += 1
    if uses_import:
        score += 1
    if uses_correct_module:
        score += 2
    if has_method_call:
        score += 1

    return {
        "uses_pyrsm": uses_pyrsm,
        "uses_import": uses_import,
        "uses_correct_module": uses_correct_module,
        "has_method_call": has_method_call,
        "score": score,
        "max_score": 5
    }

def test_prompt(model, prompt: str, context: str = "") -> dict:
    """Test a single prompt with optional context"""

    system_prompt = "You are helping a student write Python code for data analysis. Generate concise, working code."

    if context:
        full_prompt = f"{system_prompt}\n\nYou have access to this documentation:\n{context}\n\nStudent request: {prompt}"
    else:
        full_prompt = f"{system_prompt}\n\nStudent request: {prompt}"

    response = model.generate_content(full_prompt)
    response_text = response.text

    scores = score_response(response_text)

    return {
        "prompt": prompt,
        "response": response_text,
        **scores
    }

def run_test_suite(model, context_name: str, context: str):
    """Run all tests with given context"""
    print(f"\n{'='*80}")
    print(f"Testing with: {context_name}")
    print('='*80)

    results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] {prompt[:60]}...")
        result = test_prompt(model, prompt, context)
        results.append(result)
        print(f"  Score: {result['score']}/{result['max_score']} | Uses pyrsm: {result['uses_pyrsm']}")

    return results

def print_summary(all_results: dict):
    """Print comparison summary"""
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print('='*80)

    for context_name, results in all_results.items():
        total_score = sum(r['score'] for r in results)
        max_possible = sum(r['max_score'] for r in results)
        pyrsm_count = sum(1 for r in results if r['uses_pyrsm'])

        print(f"\n{context_name}:")
        print(f"  Total Score: {total_score}/{max_possible} ({100*total_score/max_possible:.1f}%)")
        print(f"  Used pyrsm: {pyrsm_count}/{len(results)} ({100*pyrsm_count/len(results):.1f}%)")

def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in ~/.env")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')  # Fast and cost-effective

    print("AUTOMATED DOCUMENTATION TEST")
    print(f"Testing {len(TEST_PROMPTS)} prompts with 3 different contexts\n")

    # Load contexts
    readme = load_readme()
    examples = load_example_context()

    # Run tests
    all_results = {}
    all_results["NO CONTEXT (baseline)"] = run_test_suite(model, "NO CONTEXT (baseline)", "")
    all_results["With README"] = run_test_suite(model, "With README", readme)
    all_results["With Examples"] = run_test_suite(model, "With Examples", examples)

    # Print summary
    print_summary(all_results)

    print(f"\n{'='*80}")
    print("DETAILED EXAMPLE (First prompt with each context):")
    print('='*80)
    for context_name, results in all_results.items():
        print(f"\n{context_name}:")
        print(results[0]['response'][:300] + "...")

if __name__ == "__main__":
    main()
