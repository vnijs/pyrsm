# PYRSM MCP Server

This repository includes an MCP (Model Context Protocol) server for the PYRSM package, enabling easy interaction with the statistical analysis functionality through tools like GitHub Copilot Chat in VS Code.

## Features

The MCP server currently provides the following tools:

1. **get_available_datasets**: Lists all available datasets in the PYRSM package
2. **load_dataset**: Loads a specific dataset by name
3. **generate_sample_data**: Creates sample data for statistical analysis
4. **single_mean_test**: Performs a single mean hypothesis test
5. **analyze_data**: Analyzes a dataset based on a natural language question

## Setup

### Prerequisites

- VS Code with GitHub Copilot and GitHub Copilot Chat extensions
- Python 3.12+
- UV package manager

### Installation

1. Install the required packages using UV:

```bash
uv add "mcp[cli]"
```

2. Install PYRSM in development mode:

```bash
uv pip install -e .
```

3. Create a `.vscode/mcp.json` file in your project directory with the following content:

```json
{
  "servers": {
    "PYRSM Statistical Analysis": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "mcp", "dev", "mcp_server.py"],
      "description": "MCP server for statistical analysis with PYRSM"
    }
  }
}
```

4. Open the project in VS Code

## Usage

### Starting the Server

There are two ways to start the MCP server:

1. **Through VS Code**:
   - VS Code will automatically detect the MCP server configuration in `.vscode/mcp.json`
   - Open GitHub Copilot Chat in VS Code
   - Enable the Agent mode (click on the robot icon in the chat input)
   - Choose "PYRSM Statistical Analysis" from the available MCP servers when prompted

2. **Manually**:
   - Run the following command in the terminal:
   ```bash
   uv run mcp dev mcp_server.py
   ```

### Using with GitHub Copilot Chat

Once the server is running, you can interact with it through GitHub Copilot Chat in VS Code.

Here are some example prompts you can use:

1. "Which datasets are available?"

2. "Load the demand_uk dataset"

3. "Is the demand column larger than 1750?"

4. "Generate a sample dataset with 100 observations, a mean of 75, and a standard deviation of 10."

5. "Perform a single mean test to determine if the mean of the values is greater than 95."

6. "Help me interpret the results of the single mean test."

### Jupyter Notebook Integration

The MCP server includes special support for Jupyter notebooks in VS Code, allowing you to:

1. List available datasets: Ask "Which datasets are available?"
2. Select a dataset: Say "I want to use the [dataset name] dataset"
3. Analyze data: Ask questions like "Is the demand larger than 1750?"

The server will generate code to be inserted into your Jupyter notebook cells, run the code, and show the results.

#### Jupyter Notebook Example Interactions

**Finding Available Datasets**
```
Which datasets are available?
```

**Loading a Specific Dataset**
```
Load the demand_uk dataset
```

**Analyzing Data**
```
Is the "demand" column larger than 1750?
```

```
What's the correlation between price and demand?
```

### Example Workflow

1. Start a conversation with GitHub Copilot Chat
2. Enable the Agent mode (click on the robot icon in the chat input)
3. Select the PYRSM MCP server from the available tools
4. Ask Copilot to list available datasets:
   ```
   Which datasets are available?
   ```
5. Copilot will use the `get_available_datasets` tool to list the datasets
6. Ask Copilot to load a specific dataset:
   ```
   Load the demand_uk dataset
   ```
7. Ask Copilot to analyze the data:
   ```
   Is the demand larger than 1750?
   ```
8. Copilot will generate and run code to answer your question, showing the results directly in your notebook

## Setting Up VS Code Integration

The PYRSM MCP server can be easily integrated with VS Code's native MCP support:

1. Create a `.vscode/mcp.json` file in your project with the following content:

```json
{
  "servers": {
    "PYRSM Statistical Analysis": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "mcp", "dev", "mcp_server.py"],
      "description": "MCP server for statistical analysis with PYRSM"
    }
  }
}
```

2. Open VS Code Command Palette (Ctrl+Shift+P / Cmd+Shift+P) and type:
   ```
   MCP: List Servers
   ```

3. You should see "PYRSM Statistical Analysis" in the list of available servers

4. In GitHub Copilot Chat, click on the robot icon to enable Agent mode, then select the PYRSM server from the list of available tools

## Extending the MCP Server

You can extend the server by adding more tools from the PYRSM package. To add a new tool:

1. Add a new function decorator with `@mcp.tool()` in `mcp_server.py`
2. Define the function with appropriate parameters and return values
3. Restart the MCP server

## Troubleshooting

If you encounter issues:

1. Make sure the MCP server is running
2. Check that VS Code can find the server configuration
   - Use Command Palette and "MCP: List Servers" to verify
3. Verify that all required packages are installed
4. Check the terminal output for any error messages from the MCP server
5. For Jupyter notebook integration, ensure that the notebook is open and cell execution is working
6. If VS Code doesn't detect the MCP server, try restarting VS Code