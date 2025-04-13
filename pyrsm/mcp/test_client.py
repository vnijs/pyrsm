"""
Test client for the pyrsm MCP server.
"""

import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def test_single_mean():
    """Test the single_mean functionality."""
    from pyrsm.mcp.single_mean_server import handle_request
    
    # Create a test request with sample data
    np.random.seed(42)
    test_data = pd.DataFrame({
        "values": np.random.normal(100, 15, 50)
    })
    
    request = {
        "type": "test",
        "data": {
            "data": test_data.to_dict(),
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
    print("Response status:", response["status"])
    if response["status"] == "success":
        print("\nTest Results:")
        print(f"Mean: {response['data']['mean']}")
        print(f"t-value: {response['data']['t_value']}")
        print(f"p-value: {response['data']['p_value']}")
        print(f"Confidence Interval: {response['data']['confidence_interval']}")
        print(f"Sample Size: {response['data']['sample_size']}")
        
        print("\nSummary Text:")
        print(response['data']['summary_text'])
        
        # Display the plot
        if 'plot' in response['data']:
            img_data = base64.b64decode(response['data']['plot'])
            img = BytesIO(img_data)
            plt.figure(figsize=(10, 6))
            plt.imshow(plt.imread(img))
            plt.axis('off')
            plt.title("Single Mean Test Plot")
            plt.show()
    else:
        print("Error:", response.get("error", "Unknown error"))

def test_sample_data():
    """Test the sample data functionality."""
    from pyrsm.mcp.single_mean_server import handle_request
    
    # Create a request for sample data
    request = {
        "type": "sample_data"
    }
    
    # Send request and get response
    response_str = handle_request(json.dumps(request))
    response = json.loads(response_str)
    
    # Print the response
    print("Response status:", response["status"])
    if response["status"] == "success":
        # Convert the data back to a DataFrame
        df = pd.DataFrame(response["data"])
        print("\nSample Data:")
        print(df.head())
        
        # Display a histogram of the data
        plt.figure(figsize=(10, 6))
        df["values"].hist()
        plt.title("Sample Data Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("Error:", response.get("error", "Unknown error"))

if __name__ == "__main__":
    print("Testing single_mean functionality...")
    test_single_mean()
    
    print("\nTesting sample_data functionality...")
    test_sample_data()