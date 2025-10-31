# Quick Start - Testing the MCP Server

## ✅ Setup Complete!

I've configured the MCP server for BOTH:
- **Claude Code**: via `.mcp.json` in this directory
- **VS Code**: via `~/.config/Code/User/mcp.json`

## How to Test

### Option 1: Test in Claude Code (this session)

1. **Restart Claude Code** (close and reopen completely)

2. **Start a NEW conversation** (important - this conversation won't have the tool)

3. **Ask**:

What MCP tools do you have?

4. **You should see**: `single_mean_test` tool

5. **Try it**:

Test if the mean of price is 100

### Option 2: Test in VS Code with MCP Extension

1. Make sure you have the **MCP extension** installed in VS Code

2. Open VS Code in this directory:
   ```bash
   cd /home/vnijs/gh/pyrsm/mcp-server
   code .
   ```

3. Check the MCP panel (should see "pyrsm" server)

4. Use the AI assistant and ask about testing means

## Expected Behavior

When working, you'll see:
```
Single mean test
Data      : Not provided
Variables : price
Confidence: 0.95
Comparison: 100

 mean  n  n_missing    sd    se    me
101.8 10          0 4.826 1.526 3.452
 diff    se  t.value  p.value  df   2.5%   97.5%
  1.8 1.526     1.18    0.268   9 98.348 105.252
```

## Troubleshooting

If it doesn't work:

1. **Check the server starts**:
   ```bash
   cd /home/vnijs/gh/pyrsm/mcp-server
   source .venv/bin/activate
   python server.py
   ```

   Should show JSON-RPC error (this is normal - needs a client)

2. **Check paths are correct**:
   - Python: `/home/vnijs/gh/pyrsm/mcp-server/.venv/bin/python`
   - Server: `/home/vnijs/gh/pyrsm/mcp-server/server.py`

3. **Restart everything** - MCP servers load on startup

## What's Next?

Once this ONE tool works, we can easily add:
- `compare_means` - Compare two groups
- `regress` - Linear regression
- `logistic` - Logistic regression
- `rforest` - Random forest
- Load actual data files
- Generate AND execute code

Try it now! 🚀
