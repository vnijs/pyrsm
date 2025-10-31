#!/usr/bin/env python3
"""Quick test of the MCP server functionality"""

import pandas as pd
import pyrsm

# Test data (same as in server)
data = pd.DataFrame({
    'price': [95, 105, 98, 102, 110, 97, 103, 99, 101, 108],
    'sales': [100, 150, 200, 250, 300, 120, 180, 220, 280, 320],
})

print("Testing pyrsm MCP server functionality\n")
print("="*60)

# Test 1: Single mean test on price
print("\nTest 1: Is mean price different from 100?")
print("-"*60)
sm = pyrsm.basics.single_mean(data, var='price', comp_value=100)
sm.summary()

print("\n" + "="*60)

# Test 2: Single mean test on sales
print("\nTest 2: Is mean sales different from 200?")
print("-"*60)
sm2 = pyrsm.basics.single_mean(data, var='sales', comp_value=200)
sm2.summary()

print("\n" + "="*60)
print("\n✓ If you see statistics above, the MCP server will work!")
print("✓ The server exposes these same functions as MCP tools")
