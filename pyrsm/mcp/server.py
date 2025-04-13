"""
MCP server for pyrsm package.
This is the main entry point for the MCP server.
"""

import json
import logging
import sys

from pyrsm.mcp.notebook_server import handle_notebook_request
from pyrsm.mcp.single_mean_server import handle_request

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pyrsm_mcp")

def process_line(line):
    """Process a single line of input from stdin."""
    try:
        # Parse the request to determine the type
        request = json.loads(line)
        
        # Check if this is a notebook request
        if request.get("context", {}).get("jupyter_notebook") is True:
            # Handle jupyter notebook request
            response = handle_notebook_request(line)
        else:
            # Handle regular request
            response = handle_request(line)
            
        # Write response to stdout
        sys.stdout.write(response + "\n")
        sys.stdout.flush()
    except Exception as e:
        error_response = json.dumps({
            "status": "error",
            "error": f"Server error: {str(e)}"
        })
        sys.stdout.write(error_response + "\n")
        sys.stdout.flush()
        logger.error(f"Error processing request: {str(e)}")

def run_server():
    """Run the MCP server, reading from stdin and writing to stdout."""
    logger.info("Starting pyrsm MCP server")
    
    try:
        # Read stdin line by line
        for line in sys.stdin:
            line = line.strip()
            if line:
                process_line(line)
    except KeyboardInterrupt:
        logger.info("Server shutting down")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_server()