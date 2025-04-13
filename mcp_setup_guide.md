# PYRSM MCP Server Setup Guide

This guide explains how to set up and use the PYRSM MCP server for VS Code Copilot Chat integration.

## What's Included

1. **MCP Server** (`mcp_server.py`): The main server file that defines the tools for statistical analysis.
   - `get_available_datasets`: Lists all available datasets in the PYRSM package
   - `load_dataset`: Loads a specific dataset by name
   - `generate_sample_data`: Creates sample data for statistical analysis
   - `single_mean_test`: Performs a single mean hypothesis test
   - `analyze_data`: Analyzes a dataset based on a natural language question

2. **VS Code Configuration** (`.vscode/mcp.json`): Configuration file that tells VS Code how to run the MCP server.

3. **Documentation**:
   - `MCP_README.md`: General information about the MCP server
   - `mcp_examples.md`: Example prompts and expected interactions
   - `mcp_setup_guide.md`: This setup guide

## Setup Instructions

### Prerequisites

1. Install VS Code with GitHub Copilot and GitHub Copilot Chat extensions
2. Make sure Python 3.12+ is installed
3. Have the UV package manager installed

### Installation Steps

1. Install the MCP SDK:
   ```bash
   uv add "mcp[cli]"
   ```

2. Clone the PYRSM repository if you haven't already:
   ```bash
   git clone https://github.com/vnijs/pyrsm.git
   cd pyrsm
   ```

3. Install PYRSM in development mode:
   ```bash
   uv pip install -e .
   ```

4. Verify the server works:
   ```bash
   uv run python -c "from mcp_server import get_available_datasets; print(get_available_datasets()['count'])"
   ```

## Using the MCP Server with VS Code

1. Open the PYRSM project in VS Code:
   ```bash
   code .
   ```

2. Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P) and type:
   ```
   MCP: List Servers
   ```

3. VS Code should detect the configuration in `.vscode/mcp.json` automatically.

4. Open GitHub Copilot Chat:
   - Click the Copilot Chat icon in the sidebar, or
   - Press Ctrl+Shift+I / Cmd+Shift+I

5. Enable Agent mode in Copilot Chat:
   - Click the robot icon in the chat input box
   - Select "PYRSM Statistical Analysis" from the list of available tools

6. Start using the tools with prompts like those found in `mcp_examples.md`

## Jupyter Notebook Integration

The MCP server includes special support for Jupyter notebooks in VS Code, allowing you to:

1. With a Jupyter notebook open in VS Code, enable the Copilot Chat Agent mode and select the PYRSM server

2. Ask about available datasets:
   ```
   Which datasets are available?
   ```

3. Load a dataset:
   ```
   Load the demand_uk dataset
   ```

4. Analyze data:
   ```
   Is the demand greater than 1750?
   ```

The server will generate Python code for each request, insert the code into a notebook cell, and execute the cell to show the results.

## Running the Server Manually

If you prefer to run the server manually:

```bash
uv run mcp dev mcp_server.py
```

This will start the MCP Inspector web interface, where you can test the tools directly.

## Troubleshooting

- If VS Code doesn't detect the MCP server, try restarting VS Code
- If you get import errors, make sure PYRSM is installed correctly with `uv pip install -e .`
- Check the terminal output for any error messages from the MCP server
- Make sure you're using Python 3.12+ as specified in the project requirements
- For Jupyter notebook integration, ensure the notebook is open and cell execution is working

## Extending the Server

To add more tools:

1. Open `mcp_server.py`
2. Add new functions with the `@mcp.tool()` decorator
3. Restart the MCP server to load the new tools

## Example: Full Workflow

1. Start VS Code with the PYRSM project
2. Enable the PYRSM MCP server in Copilot Chat (Agent mode)
3. Ask Copilot to list available datasets:
   ```
   Which datasets are available in PYRSM?
   ```
4. Ask Copilot to load a dataset:
   ```
   Load the demand_uk dataset
   ```
5. Ask Copilot to analyze the data:
   ```
   Is the demand larger than 1750?
   ```
6. Explore the dataset with more questions:
   ```
   What's the correlation between the columns?
   ```