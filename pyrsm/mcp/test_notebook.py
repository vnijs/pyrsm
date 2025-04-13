"""
Test client for testing the Jupyter notebook integration with the PYRSM MCP server.
"""

import json
import sys
from pyrsm.mcp.notebook_server import handle_notebook_request

def test_get_available_datasets():
    """Test retrieving available datasets."""
    request = {
        "type": "get_available_datasets",
        "data": {},
        "context": {"jupyter_notebook": True}
    }
    
    print("Testing get_available_datasets...")
    response_str = handle_notebook_request(json.dumps(request))
    response = json.loads(response_str)
    
    print("Response status:", response.get("status"))
    if response.get("status") == "success":
        print(f"Found {len(response.get('data', []))} datasets")
        
        # Print the generated code
        print("\nGenerated code:")
        print(response.get("code", "No code generated"))
    else:
        print("Error:", response.get("error"))
    
    return response

def test_load_dataset(dataset_name="demand_uk"):
    """Test loading a specific dataset."""
    request = {
        "type": "load_dataset",
        "data": {
            "dataset_name": dataset_name
        },
        "context": {"jupyter_notebook": True}
    }
    
    print(f"\nTesting load_dataset with dataset '{dataset_name}'...")
    response_str = handle_notebook_request(json.dumps(request))
    response = json.loads(response_str)
    
    print("Response status:", response.get("status"))
    if response.get("status") == "success":
        data = response.get("data", {})
        print(f"Dataset: {data.get('name')}")
        print(f"Shape: {data.get('rows')} rows x {data.get('columns')} columns")
        print(f"Columns: {data.get('column_names')}")
        
        # Print the generated code
        print("\nGenerated code:")
        print(response.get("code", "No code generated"))
    else:
        print("Error:", response.get("error"))
    
    return response

def test_analyze_data(dataset_name="demand_uk", question="Is demand larger than 1750?"):
    """Test analyzing data with a natural language question."""
    request = {
        "type": "analyze_data",
        "data": {
            "dataset_name": dataset_name,
            "question": question
        },
        "context": {"jupyter_notebook": True}
    }
    
    print(f"\nTesting analyze_data with dataset '{dataset_name}' and question '{question}'...")
    response_str = handle_notebook_request(json.dumps(request))
    response = json.loads(response_str)
    
    print("Response status:", response.get("status"))
    if response.get("status") == "success":
        data = response.get("data", {})
        print(f"Explanation: {data.get('explanation')}")
        
        # Print the generated code
        print("\nGenerated code:")
        print(response.get("code", "No code generated"))
    else:
        print("Error:", response.get("error"))
    
    return response

def run_all_tests():
    """Run all tests for Jupyter notebook integration."""
    print("Running all Jupyter notebook integration tests...\n")
    
    # Run the tests, but we don't need to store the results
    test_get_available_datasets()
    test_load_dataset()
    
    # Test a few different types of analysis questions
    test_analyze_data(question="Is demand larger than 1750?")
    test_analyze_data(question="What are the mean, median, and standard deviation?")
    test_analyze_data(question="Show the correlation between price and demand")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    # Check if a specific test was requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "datasets":
            test_get_available_datasets()
        elif test_name == "load":
            dataset = sys.argv[2] if len(sys.argv) > 2 else "demand_uk"
            test_load_dataset(dataset)
        elif test_name == "analyze":
            dataset = sys.argv[2] if len(sys.argv) > 2 else "demand_uk"
            question = sys.argv[3] if len(sys.argv) > 3 else "Is demand larger than 1750?"
            test_analyze_data(dataset, question)
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: datasets, load, analyze")
    else:
        # Run all tests
        run_all_tests()