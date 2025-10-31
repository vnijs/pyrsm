#!/usr/bin/env python3
"""
Minimal pyrsm MCP Server - Test Version
Single tool: run a single mean test
"""

import asyncio
import sys
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import pandas as pd
import pyrsm

# Create server
app = Server("pyrsm-mcp-demo")

# Sample data for testing
SAMPLE_DATA = pd.DataFrame({
    'price': [95, 105, 98, 102, 110, 97, 103, 99, 101, 108],
    'sales': [100, 150, 200, 250, 300, 120, 180, 220, 280, 320],
})

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="single_mean_test",
            description="Test if the mean of a variable is significantly different from a comparison value. Returns summary statistics and test results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Variable name to test (e.g., 'price', 'sales')",
                        "enum": ["price", "sales"]
                    },
                    "comparison_value": {
                        "type": "number",
                        "description": "Value to compare the mean against",
                        "default": 0
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level (0-1)",
                        "default": 0.95
                    }
                },
                "required": ["variable", "comparison_value"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""

    if name == "single_mean_test":
        var = arguments["variable"]
        comp_value = arguments["comparison_value"]
        conf = arguments.get("confidence", 0.95)

        # Run pyrsm single_mean test
        sm = pyrsm.basics.single_mean(
            SAMPLE_DATA,
            var=var,
            comp_value=comp_value,
            conf=conf
        )

        # Capture summary output
        import io
        from contextlib import redirect_stdout

        output = io.StringIO()
        with redirect_stdout(output):
            sm.summary()

        result = output.getvalue()

        return [TextContent(
            type="text",
            text=f"Single Mean Test Results:\n\n{result}\n\nData used: Sample dataset with {len(SAMPLE_DATA)} observations"
        )]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
