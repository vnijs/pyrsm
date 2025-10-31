# Setup MCP Server for Claude Code

## Step 1: Find Your Claude Code Config File

The config location depends on your setup:
- `~/.config/claude-code/config.json` (Linux)
- `~/.config/claude/config.json` (alternative)

Let's check which one exists.

## Step 2: Add MCP Server Configuration

Add this to your Claude Code config file:

```json
{
  "mcpServers": {
    "pyrsm": {
      "command": "/home/vnijs/gh/pyrsm/mcp-server/.venv/bin/python",
      "args": ["/home/vnijs/gh/pyrsm/mcp-server/server.py"],
      "env": {}
    }
  }
}
```

If the file already has other content, just add the `"pyrsm"` entry inside the `"mcpServers"` object.

## Step 3: Restart Claude Code

Close and reopen Claude Code completely.

## Step 4: Test It

In a NEW Claude Code session, ask:
- "What MCP tools do you have available?"
- "Test if the mean of price is 100"

You should see the tool get called!

## Troubleshooting

If it doesn't work:
1. Check the config file exists and has correct paths
2. Make sure the virtual environment is at `/home/vnijs/gh/pyrsm/mcp-server/.venv`
3. Check Claude Code logs for errors
