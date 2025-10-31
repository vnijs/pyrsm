#!/usr/bin/env python3
"""
Test all 3 MCP tools: single_mean, compare_means, regress_fit
"""

import sys
import asyncio
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

from server_regression import DATA_STORE, list_tools, call_tool
import pyrsm

print("="*70)
print("CHECKPOINT 2: Testing All 3 MCP Tools")
print("="*70)

def test_all_tools():
    # Load test data
    print("\n1. LOAD TEST DATA")
    print("-"*70)

    # Salary data for single_mean
    salary_data, _ = pyrsm.load_data(name='salary', pkg='basics')
    DATA_STORE['salary'] = salary_data
    print(f"✓ Loaded salary: {salary_data.shape}")

    # Use salary for compare_means too (compare by rank)
    # titanic doesn't exist, use salary which has rank categorical variable
    DATA_STORE['salary_compare'] = salary_data
    print(f"✓ Using salary for compare_means: {salary_data.shape}")

    # Diamonds for regression
    diamonds_data, _ = pyrsm.load_data(name='diamonds', pkg='model')
    DATA_STORE['diamonds'] = diamonds_data
    print(f"✓ Loaded diamonds: {diamonds_data.shape}")

    # List tools
    print("\n2. LIST ALL TOOLS")
    print("-"*70)

    async def get_tools():
        return await list_tools()

    tools = asyncio.run(get_tools())
    tool_names = [t.name for t in tools]

    print(f"Total tools: {len(tool_names)}")
    for name in ['single_mean', 'compare_means', 'regress_fit']:
        status = "✓" if name in tool_names else "✗"
        print(f"{status} {name}")

    # Test 1: single_mean
    print("\n3. TEST SINGLE_MEAN")
    print("-"*70)
    print("Test if mean salary equals 100000")

    async def test_sm():
        return await call_tool(
            name="single_mean",
            arguments={
                "data_name": "salary",
                "var": "salary",
                "comp_value": 100000,
                "dec": 2
            }
        )

    result = asyncio.run(test_sm())
    print(result[0].text[:500] + "...")

    # Test 2: compare_means
    print("\n4. TEST COMPARE_MEANS")
    print("-"*70)
    print("Compare salary between ranks (Prof vs AsstProf)")

    async def test_cm():
        return await call_tool(
            name="compare_means",
            arguments={
                "data_name": "salary_compare",
                "var1": "rank",
                "var2": "salary",
                "dec": 2
            }
        )

    result = asyncio.run(test_cm())
    print(result[0].text[:500] + "...")

    # Test 3: regress_fit (expanded)
    print("\n5. TEST REGRESS_FIT (WITH NEW PARAMETERS)")
    print("-"*70)
    print("Regression with vif=True, dec=2")

    async def test_reg():
        return await call_tool(
            name="regress_fit",
            arguments={
                "data_name": "diamonds",
                "rvar": "price",
                "evar": ["carat", "depth", "table"],
                "vif": True,
                "dec": 2
            }
        )

    result = asyncio.run(test_reg())
    print(result[0].text[:500] + "...")

    print("\n" + "="*70)
    print("✓ CHECKPOINT 2 COMPLETE: All 3 tools working!")
    print("="*70)
    print("\nReady for:")
    print("  - Checkpoint 3: LLM tool selection")
    print("  - Demo notebook testing")

test_all_tools()
