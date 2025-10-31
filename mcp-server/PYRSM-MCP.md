# pyrsm MCP Server

Model Context Protocol (MCP) server that exposes pyrsm statistical functions as tools for AI assistants.

## Quick Start

```bash
cd /home/vnijs/gh/pyrsm/mcp-server

# Install dependencies
make install

# Test functionality
make test

# Verify server starts
make verify
```

## Configuration

### For VS Code
```bash
make config-vscode
```

Creates `~/.config/Code/User/mcp.json`:
```json
{
  "mcpServers": {
    "pyrsm": {
      "command": "/home/vnijs/gh/pyrsm/mcp-server/.venv/bin/python",
      "args": ["/home/vnijs/gh/pyrsm/mcp-server/server.py"]
    }
  }
}
```

**Restart VS Code** to activate.

### For Claude Code
```bash
make config-claude
```

Creates `.mcp.json` in this directory.

**Restart Claude Code** to activate.

## Available Tools

### `single_mean_test`
Test if the mean of a variable is significantly different from a comparison value.

**Parameters:**
- `variable` (required): Variable to test ('price' or 'sales')
- `comparison_value` (required): Value to compare against
- `confidence` (optional): Confidence level (default: 0.95)

**Example prompts:**
- "Test if the mean of price is different from 100"
- "Is the average sales significantly different from 200?"

## Testing

### Run Functional Tests
```bash
make test
```

Output shows two hypothesis tests:
```
Single mean test
Variables : price
Comparison: 100

 mean  n  n_missing    sd    se    me
101.8 10          0 4.826 1.526 3.452
```

### Start Server Manually
```bash
make run
```

Expect to see JSON-RPC errors - this is normal without a client connected.

### Test with MCP Inspector

Install MCP Inspector:
```bash
npx @modelcontextprotocol/inspector /home/vnijs/gh/pyrsm/mcp-server/.venv/bin/python /home/vnijs/gh/pyrsm/mcp-server/server.py
```

Opens a web UI to:
- List available tools
- Test tool calls with parameters
- View responses

### Test via API Call

Direct JSON-RPC call (for debugging):

```bash
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | \
  .venv/bin/python server.py
```

Expected response includes `single_mean_test` tool definition.

## Using in AI Assistants

### Claude Code
Start a new conversation and ask:
```
What MCP tools do you have available?
```

Should list `single_mean_test`.

Then try:
```
Test if the mean of price is 100
```

### VS Code with GitHub Copilot Chat
With MCP extension installed:
```
@mcp Test the mean of price against 100
```

## Sample Data

The server uses built-in sample data:
```python
{
    'price': [95, 105, 98, 102, 110, 97, 103, 99, 101, 108],
    'sales': [100, 150, 200, 250, 300, 120, 180, 220, 280, 320]
}
```

10 observations, suitable for t-tests.

## Architecture

```
server.py
├── @app.list_tools()      # Returns available tools
├── @app.call_tool()       # Executes tool with parameters
└── Uses pyrsm.basics.single_mean() internally
```

**Flow:**
1. AI assistant requests available tools
2. Server returns `single_mean_test` schema
3. AI decides to call tool with parameters
4. Server executes `pyrsm.basics.single_mean()`
5. Returns formatted results to AI

## Development

### Add New Tools

Edit `server.py`:

```python
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="compare_means",
            description="Compare means between two groups",
            inputSchema={
                "type": "object",
                "properties": {
                    "var1": {"type": "string"},
                    "var2": {"type": "string"}
                },
                "required": ["var1", "var2"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "compare_means":
        # Execute pyrsm.basics.compare_means()
        pass
```

### Test Changes
```bash
make verify
```

## Troubleshooting

**Server won't start:**
```bash
# Check Python version (needs >=3.12)
python --version

# Reinstall
make clean
make install
```

**Tools not appearing in AI:**
- Restart the AI assistant completely
- Check config files exist:
  ```bash
  cat ~/.config/Code/User/mcp.json
  cat .mcp.json
  ```

**JSON-RPC errors when running directly:**
- This is expected - the server needs an MCP client
- Use `make test` to verify functionality instead

## Next Steps

1. **Test the basic setup** - Use prompts above
2. **Add more tools** - Expose other pyrsm functions
3. **Load real data** - Replace sample data with file uploads
4. **Generate code mode** - Return executable code instead of results
5. **Quarto integration** - Insert code + results into .qmd files

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [pyrsm Documentation](https://github.com/vnijs/pyrsm)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
