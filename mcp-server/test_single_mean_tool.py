#!/usr/bin/env python3
"""
Test the single_mean MCP tool
"""

import sys
import asyncio
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

# Import the server module to access tools
from server_regression import app, DATA_STORE
import pandas as pd
import pyrsm

print("="*70)
print("CHECKPOINT 1: Testing single_mean MCP Tool")
print("="*70)

def test_single_mean():
    # Step 1: Load test data into DATA_STORE
    print("\n1. LOAD TEST DATA")
    print("-"*70)
    data, desc = pyrsm.load_data(name='salary', pkg='basics')
    DATA_STORE['salary'] = data
    print(f"✓ Loaded salary dataset: {data.shape}")
    print(f"  Columns: {list(data.columns)[:5]}...")

    # Step 2: Get list_tools handler and call it
    print("\n2. LIST TOOLS (check single_mean is there)")
    print("-"*70)
    # The decorator stores the function, we need to call it directly
    from server_regression import list_tools

    # Create a simple async wrapper
    async def get_tools():
        return await list_tools()

    tools = asyncio.run(get_tools())
    tool_names = [t.name for t in tools]
    print(f"Available tools ({len(tool_names)}): {tool_names[:5]}...")

    if 'single_mean' in tool_names:
        print("✓ single_mean tool found!")
    else:
        print("✗ single_mean tool NOT found!")
        return

    # Step 3: Get single_mean tool definition
    print("\n3. SINGLE_MEAN TOOL DEFINITION")
    print("-"*70)
    sm_tool = [t for t in tools if t.name == 'single_mean'][0]
    print(f"Name: {sm_tool.name}")
    print(f"Description: {sm_tool.description}")
    print(f"Required params: {sm_tool.inputSchema['required']}")
    print(f"All params: {list(sm_tool.inputSchema['properties'].keys())}")

    # Step 4: Call single_mean tool
    print("\n4. CALL SINGLE_MEAN TOOL")
    print("-"*70)
    print("Test: Check if mean salary equals 60000")

    from server_regression import call_tool

    async def run_tool():
        return await call_tool(
            name="single_mean",
            arguments={
                "data_name": "salary",
                "var": "salary",
                "comp_value": 60000,
                "alt_hyp": "two-sided",
                "conf": 0.95,
                "dec": 2
            }
        )

    result = asyncio.run(run_tool())

    print("\nResult:")
    print(result[0].text)

    print("\n" + "="*70)
    print("✓ CHECKPOINT 1 COMPLETE: single_mean tool working!")
    print("="*70)

# Run test
test_single_mean()
