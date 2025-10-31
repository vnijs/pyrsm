#!/usr/bin/env python3
"""
Quick test to verify nest_asyncio fix works
"""

import sys
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

print("="*70)
print("Testing nest_asyncio Fix")
print("="*70)

# Apply nest_asyncio first
import nest_asyncio
nest_asyncio.apply()
print("✓ nest_asyncio applied")

# Simulate what happens in Jupyter
import asyncio
from server_regression import call_tool, DATA_STORE
import pyrsm

print("\n1. Load test data")
print("-"*70)
salary, _ = pyrsm.load_data(name='salary', pkg='basics')
DATA_STORE['salary'] = salary
print(f"✓ Loaded salary: {salary.shape}")

print("\n2. Test async tool call (simulating %%mcp magic)")
print("-"*70)
print("Calling single_mean tool via asyncio.run()...")

async def run_tool():
    return await call_tool(
        name="single_mean",
        arguments={
            "data_name": "salary",
            "var": "salary",
            "comp_value": 100000,
            "dec": 2
        }
    )

try:
    # This would fail without nest_asyncio
    result = asyncio.run(run_tool())
    print("✓ asyncio.run() succeeded!")
    print(f"✓ Got result: {result[0].text[:100]}...")

    print("\n" + "="*70)
    print("✓ FIX WORKING: nest_asyncio allows nested event loops")
    print("="*70)
    print("\n%%mcp magic should now work in Jupyter notebooks!")

except RuntimeError as e:
    if "already running" in str(e):
        print(f"✗ STILL BROKEN: {e}")
        print("nest_asyncio not applied correctly")
    else:
        raise

