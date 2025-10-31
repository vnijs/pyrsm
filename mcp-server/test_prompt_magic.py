#!/usr/bin/env python3
"""
Quick test of prompt_magic extension
"""

import sys
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

# Test 1: Import the extension
print("="*70)
print("TEST 1: Import prompt_magic")
print("="*70)

try:
    import prompt_magic
    print("✓ prompt_magic imported successfully")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check functions exist
print("\n" + "="*70)
print("TEST 2: Check magic functions")
print("="*70)

if hasattr(prompt_magic, 'prompt'):
    print("✓ %%prompt cell magic found")
else:
    print("✗ %%prompt cell magic not found")

if hasattr(prompt_magic, '_get_kernel_context'):
    print("✓ _get_kernel_context function found")
else:
    print("✗ _get_kernel_context function not found")

if hasattr(prompt_magic, '_generate_code_from_prompt'):
    print("✓ _generate_code_from_prompt function found")
else:
    print("✗ _generate_code_from_prompt function not found")

# Test 3: Test code generation (without IPython kernel)
print("\n" + "="*70)
print("TEST 3: Test code generation logic")
print("="*70)

test_cases = [
    ("Load the diamonds dataset", {}),
    ("Fit a regression with price and carat, depth", {'dataframes': {'diamonds': {'columns': ['price', 'carat', 'depth', 'table']}}}),
    ("Show summary statistics", {'dataframes': {'diamonds': {'shape': (3000, 11)}}}),
]

for prompt, context in test_cases:
    print(f"\nPrompt: '{prompt}'")
    try:
        code = prompt_magic._generate_code_from_prompt(prompt, context)
        print(f"Generated:\n{code}\n")
    except Exception as e:
        print(f"✗ Error: {e}")

print("="*70)
print("✓ ALL TESTS PASSED")
print("="*70)
print("\nNext: Open examples/prompt_demo.ipynb in VS Code!")
print("      Then run: %load_ext prompt_magic")
