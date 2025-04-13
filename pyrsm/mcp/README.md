# PYRSM MCP Server

This is a Multimodal Code Project (MCP) server for the PYRSM package, designed to work with GitHub Copilot Chat in VS Code.

## Overview

The PYRSM MCP server provides access to the statistical analysis functionality of the PYRSM package through a simple JSON API. This allows tools like GitHub Copilot Chat to interact with PYRSM and perform statistical analysis tasks.

## Current Functionality

The server currently supports:

- Listing available datasets in the PYRSM package
- Loading specific datasets for analysis
- Single mean hypothesis testing via the `single_mean` module
- Creating sample datasets for demonstration purposes
- Analyzing data with natural language queries
- Jupyter notebook integration for code generation and execution

## How to Use

### Starting the Server

To start the MCP server, run:

```bash
python -m pyrsm.mcp.server
```

This will start the server, which reads JSON requests from stdin and writes JSON responses to stdout.

Alternatively, you can use the MCP CLI:

```bash
mcp dev mcp_server.py
```

### Testing the Server

You can test the server using the included test client:

```bash
python -m pyrsm.mcp.test_client
```

### Request Format

Requests to the server should be in JSON format with the following structure:

```json
{
  "type": "request_type",
  "data": {
    // Request-specific parameters
  },
  "context": {
    // Optional context information
    "jupyter_notebook": true  // Set to true for Jupyter notebook integration
  }
}
```

Where `request_type` can be:

- `"get_available_datasets"`: List available datasets
- `"load_dataset"`: Load a specific dataset
- `"analyze_data"`: Analyze a dataset with a natural language query
- `"sample_data"`: Generate sample data
- `"test"`: Perform a single mean test

#### Jupyter Notebook Integration

For Jupyter notebook integration, include `"jupyter_notebook": true` in the context object of your request. This will trigger the server to generate code that can be inserted into notebook cells.

Example notebook request:

```json
{
  "type": "analyze_data",
  "data": {
    "dataset_name": "demand_uk",
    "question": "Is demand greater than 1750?"
  },
  "context": {
    "jupyter_notebook": true
  }
}
```

The response will include a `code` field containing Python code to run the analysis.

## Using with Jupyter Notebooks

When using the MCP server with Jupyter notebooks, the server will detect the Jupyter notebook context and generate executable code that is inserted directly into notebook cells.

### Example Workflow

1. Open a Jupyter notebook in VS Code
2. Enable the PYRSM MCP server in Copilot Chat
3. Ask about available datasets:
   ```
   Which datasets are available in PYRSM?
   ```
4. Load a dataset:
   ```
   Load the demand_uk dataset for analysis
   ```
5. Ask analysis questions:
   ```
   Is the demand greater than 1750?
   ```
   
The MCP server will generate Python code for each request, insert it into a new notebook cell, and execute the cell to show the results.

### Advanced Usage

You can also perform more complex analyses:

1. Basic statistics:
   ```
   What are the mean, median, and standard deviation of demand?
   ```

2. Correlations:
   ```
   Show me the correlation between price and demand
   ```

3. Visualizations:
   ```
   Create a histogram of the demand values
   ```

## Integration with Copilot Chat

To use this MCP server with GitHub Copilot Chat in VS Code:

1. Start the MCP server as described above
2. Open VS Code with the PYRSM project
3. Open GitHub Copilot Chat
4. Enable the PYRSM MCP server in Copilot Chat
5. Start interacting with the server through Copilot Chat

For more detailed setup instructions, see the `copilot_setup.md` file in this directory.

## Example

Here's an example of performing a single mean test:

```python
import json
from pyrsm.mcp.single_mean_server import handle_request

# Create a test request
request = {
    "type": "test",
    "data": {
        "var": "values",
        "alt_hyp": "two-sided",
        "conf": 0.95,
        "comp_value": 95
    }
}

# Send request and get response
response_str = handle_request(json.dumps(request))
response = json.loads(response_str)

# Print the response
print(json.dumps(response, indent=2))
```

## Troubleshooting

If you encounter issues:

1. Make sure the MCP server is running
2. Check that you have the necessary extensions for VS Code installed
3. Verify that the notebook kernel is running properly (for Jupyter integration)
4. Check the output from the MCP server for error messages
5. Try restarting the Jupyter kernel if you experience issues with code execution