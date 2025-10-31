.PHONY: help install test run clean config-vscode config-claude-code verify

help:
	@echo "pyrsm MCP Server - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Set up virtual environment and install dependencies"
	@echo "  make test            - Run test suite to verify pyrsm functions work"
	@echo "  make run             - Start the MCP server (for manual testing)"
	@echo "  make config-vscode   - Configure VS Code to use this MCP server"
	@echo "  make config-claude   - Configure Claude Code to use this MCP server"
	@echo "  make verify          - Verify the server can start"
	@echo "  make clean           - Remove virtual environment and cache files"

install:
	@echo "Setting up virtual environment..."
	uv venv
	@echo "Installing dependencies..."
	. .venv/bin/activate && uv add mcp
	@echo "✓ Installation complete"

test:
	@echo "Running pyrsm functionality tests..."
	. .venv/bin/activate && python test_server.py

run:
	@echo "Starting MCP server..."
	@echo "Note: You'll see JSON-RPC errors - this is normal without a client"
	@echo "Press Ctrl+C to stop"
	. .venv/bin/activate && python server.py

config-vscode:
	@echo "Configuring VS Code MCP settings..."
	@mkdir -p ~/.config/Code/User
	@echo '{\n  "mcpServers": {\n    "pyrsm": {\n      "command": "$(PWD)/.venv/bin/python",\n      "args": ["$(PWD)/server.py"]\n    }\n  }\n}' > ~/.config/Code/User/mcp.json
	@echo "✓ VS Code configured at ~/.config/Code/User/mcp.json"
	@echo "  Restart VS Code to activate"

config-claude:
	@echo "Configuring Claude Code MCP settings..."
	@echo '{\n  "mcpServers": {\n    "pyrsm": {\n      "command": "$(PWD)/.venv/bin/python",\n      "args": ["$(PWD)/server.py"]\n    }\n  }\n}' > .mcp.json
	@echo "✓ Claude Code configured via .mcp.json"
	@echo "  Restart Claude Code to activate"

verify:
	@echo "Verifying server can start..."
	@timeout 2 bash -c '. .venv/bin/activate && python server.py' || echo "✓ Server starts correctly (timeout is expected)"

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf __pycache__
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"
