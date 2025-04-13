# Setting up PYRSM MCP with GitHub Copilot Chat

This guide explains how to set up and use the PYRSM MCP server with GitHub Copilot Chat in VS Code.

## Prerequisites

- VS Code with GitHub Copilot and GitHub Copilot Chat extensions installed
- PYRSM package installed

## Step 1: Start the MCP Server

First, you need to start the MCP server in a terminal:

```bash
python -m pyrsm.mcp.server
```

Keep this terminal open while using Copilot Chat with PYRSM.

## Step 2: Configure Copilot Chat to use PYRSM MCP

Currently, GitHub Copilot Chat doesn't have a built-in way to connect to custom MCP servers. However, you can use the following workaround:

1. Create a simple script that will send requests to the MCP server and display results:

```python
# copilot_pyrsm.py
import json
import sys
import subprocess

def send_to_mcp(request_str):
    """Send a request to the MCP server via subprocess."""
    command = ["python", "-c", f"import sys; import json; from pyrsm.mcp.single_mean_server import handle_request; print(handle_request('{request_str}'))"]
    result = subprocess.run(command, text=True, capture_output=True)
    return result.stdout.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python copilot_pyrsm.py <request_json>")
        sys.exit(1)
    
    request_str = sys.argv[1]
    response = send_to_mcp(request_str)
    print(response)

if __name__ == "__main__":
    main()
```

2. Save this script in your project directory.

3. You can now invoke the PYRSM MCP from the command line:

```bash
python copilot_pyrsm.py '{"type": "sample_data"}'
```

4. In Copilot Chat, you can ask Copilot to help you generate the appropriate JSON request for your analysis needs and then use the script to send it to the MCP server.

## Step 3: Use PYRSM with Copilot Chat

Here are some example prompts you can use with Copilot Chat:

1. "Create a single mean test with PYRSM for a dataset with an expected mean of 100 and test it against a null hypothesis of 95."

2. "Generate a JSON request for PYRSM MCP to perform a one-sided hypothesis test on sample data."

3. "Help me interpret the results of a single mean test from PYRSM."

4. "What parameters can I adjust for a single mean test in PYRSM?"

## Example Flow

1. **User**: "Can you help me use PYRSM to perform a single mean test?"

2. **Copilot**: "I can help you with that. Let's create a request for the PYRSM MCP server to perform a single mean test. Here's a sample command:

   ```bash
   python copilot_pyrsm.py '{\"type\": \"test\", \"data\": {\"var\": \"values\", \"alt_hyp\": \"two-sided\", \"conf\": 0.95, \"comp_value\": 95}}'
   ```

   This will run a two-sided test comparing the mean of the 'values' column to 95 with a 95% confidence level. The server will use sample data since we didn't provide any."

3. **User**: "How do I interpret the results?"

4. **Copilot**: [Provides explanation of the statistical test results...]

## Note

This is a simple workaround until GitHub Copilot Chat provides official support for custom MCP servers. In a production environment, you would want to set up proper communication between Copilot Chat and your MCP server.