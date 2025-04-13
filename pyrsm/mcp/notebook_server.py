"""
MCP server for Jupyter notebook integration with pyrsm.
This module handles requests from Jupyter notebooks and generates code to be inserted.
"""

import json
import re

def create_response(obj, code=None, message=None, status="success", error=None):
    """Create a standardized response format for notebook integration."""
    response = {"status": status}
    
    if message:
        response["message"] = message
    
    if error:
        response["error"] = str(error)
        
    if obj is not None:
        response["data"] = obj
    
    if code is not None:
        response["code"] = code
        
    return response

def parse_input(data_str):
    """Parse the input JSON data."""
    try:
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError as e:
        return create_response(None, status="error", error=f"Invalid JSON: {str(e)}")

def get_available_datasets():
    """
    Get a list of all available datasets in the pyrsm package.
    
    Returns
    -------
    list
        A list of dictionaries containing dataset information.
    """
    from pyrsm.example_data import load_data
    
    # Get all datasets
    all_data, all_descriptions = load_data()
    
    # Format the result
    datasets = []
    
    for name, data in all_data.items():
        description = all_descriptions.get(name, "No description available")
        
        # Get just the first 100 characters of the description
        short_description = description[:100] + "..." if len(description) > 100 else description
        
        datasets.append({
            "name": name,
            "rows": len(data),
            "columns": len(data.columns),
            "description": short_description,
            "column_names": list(data.columns)
        })
    
    return datasets

def load_dataset(name):
    """
    Load a specific dataset by name.
    
    Parameters
    ----------
    name : str
        Name of the dataset to load
        
    Returns
    -------
    tuple
        (dataset, description)
    """
    from pyrsm.example_data import load_data
    
    # Try to find which package the dataset is in
    for pkg in ["data", "design", "basics", "model", "multivariate"]:
        try:
            data, description = load_data(pkg=pkg, name=name)
            return data, description
        except:
            continue
    
    # If we can't find it by package, try loading it directly
    try:
        data, description = load_data(name=name)
        return data, description
    except:
        raise ValueError(f"Dataset '{name}' not found")

def analyze_question(question, data=None):
    """
    Analyze a data analysis question and generate code to answer it.
    
    Parameters
    ----------
    question : str
        The data analysis question to answer
    data : pd.DataFrame, optional
        The dataset to analyze
        
    Returns
    -------
    tuple
        (code to run, explanation)
    """
    if data is None:
        return "# Please load a dataset first", "No dataset loaded"
    
    # Convert question to lowercase for easier matching
    q_lower = question.lower()
    
    # Check if we're asking about basic statistics
    if any(word in q_lower for word in ["mean", "average", "median", "std", "min", "max"]):
        stat_code = f"""
# Basic statistics
{data.name if hasattr(data, 'name') else 'data'}.describe()
"""
        return stat_code, "Generating basic statistics for the dataset"
    
    # Check for comparison questions
    elif "larger than" in q_lower or "greater than" in q_lower or "more than" in q_lower:
        # Extract the column and value for comparison
        parts = q_lower.split(" than ")
        if len(parts) < 2:
            return "# Could not parse comparison", "Could not understand the comparison query"
        
        value_part = parts[1].strip()
        try:
            compare_value = float(''.join(c for c in value_part if c.isdigit() or c == '.'))
        except:
            return ("# Could not extract comparison value", 
                   "Could not extract a numeric value for comparison")
        
        # Look for quoted text which might be the column name
        quoted = re.findall(r'"([^"]*)"', question) or re.findall(r"'([^']*)'", question)
        
        if quoted:
            column_name = quoted[0]
        else:
            # Try to find a column name in the question
            for col in data.columns:
                if col.lower() in q_lower:
                    column_name = col
                    break
            else:
                return ("# Could not determine which column to analyze", 
                       "Could not identify which column to analyze")
        
        comparison_code = f"""
# Check if values in '{column_name}' are greater than {compare_value}
import matplotlib.pyplot as plt

# Calculate basic statistics
column_data = {data.name if hasattr(data, 'name') else 'data'}['{column_name}']
mean_value = column_data.mean()
greater_than_count = (column_data > {compare_value}).sum()
percentage = (greater_than_count / len(column_data)) * 100

# Display results
print(f"Mean of '{column_name}': {{mean_value:.2f}}")
print(f"Number of values > {compare_value}: {{greater_than_count}} of {{len(column_data)}}")
print(f"Percentage greater than {compare_value}: {{percentage:.2f}}%")

# Create a histogram to visualize the distribution
plt.figure(figsize=(10, 6))
plt.hist(column_data, bins=20, alpha=0.7, color='skyblue')
plt.axvline(x={compare_value}, color='red', linestyle='--', 
           label=f'Threshold: {compare_value}')
plt.axvline(x=mean_value, color='green', linestyle='-', 
           label=f'Mean: {{mean_value:.2f}}')
plt.title(f"Distribution of '{column_name}'")
plt.xlabel(f'{column_name}')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
        msg = f"Analyzing if values in '{column_name}' are greater than {compare_value}"
        return comparison_code, msg
    
    # Check for correlation questions
    elif "correlation" in q_lower or "relationship" in q_lower:
        corr_code = f"""
# Correlation analysis
correlation_matrix = {data.name if hasattr(data, 'name') else 'data'}.corr()

# Display the correlation matrix
print("Correlation Matrix:")
correlation_matrix

# Visualize the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
"""
        return corr_code, "Generating correlation analysis for the dataset"
    
    # For other types of questions, provide a simple data overview
    else:
        overview_code = f"""
# Overview of the dataset
print("Dataset shape:", {data.name if hasattr(data, 'name') else 'data'}.shape)
print("\\nFirst 5 rows:")
{data.name if hasattr(data, 'name') else 'data'}.head()
"""
        return overview_code, "Providing an overview of the dataset"

def process_notebook_request(request_type, request_data):
    """
    Process a notebook request based on its type.
    
    Parameters
    ----------
    request_type : str
        The type of request
    request_data : dict
        The request data
        
    Returns
    -------
    dict
        The response
    """
    try:
        if request_type == "get_available_datasets":
            datasets = get_available_datasets()
            code = """
# List available datasets
import pyrsm as rsm

# Get information about available datasets
available_datasets = rsm.load_data()
print(f"Found {len(available_datasets[0])} datasets")

# Display the first few datasets
list(available_datasets[0].keys())[:10]  # Show first 10 dataset names
"""
            return create_response(datasets, code=code, message="Retrieved available datasets")
        
        elif request_type == "load_dataset":
            dataset_name = request_data.get("dataset_name")
            if not dataset_name:
                return create_response(None, status="error", error="Dataset name is required")
            
            try:
                data, description = load_dataset(dataset_name)
                
                # Generate code to load this dataset
                load_code = f"""
# Load the {dataset_name} dataset
import pyrsm as rsm

# Load data into the current environment
{dataset_name}, {dataset_name}_description = rsm.load_data(name="{dataset_name}")

# Display dataset information
print(f"Dataset: {dataset_name}")
print(f"Shape: {{{dataset_name}.shape}}")
print("\\nFirst 5 rows:")
{dataset_name}.head()
"""
                
                # Format the result
                result = {
                    "name": dataset_name,
                    "rows": len(data),
                    "columns": len(data.columns),
                    "column_names": list(data.columns),
                    "description": description,
                    "head": data.head().to_dict(),
                }
                
                msg = f"Loaded dataset: {dataset_name}"
                return create_response(result, code=load_code, message=msg)
            
            except Exception as e:
                return create_response(
                    None, status="error", error=f"Error loading dataset: {str(e)}"
                )
        
        elif request_type == "analyze_data":
            dataset_name = request_data.get("dataset_name")
            question = request_data.get("question")
            
            if not dataset_name:
                return create_response(None, status="error", error="Dataset name is required")
            
            if not question:
                return create_response(None, status="error", error="Analysis question is required")
            
            try:
                # Load the dataset
                data, _ = load_dataset(dataset_name)
                
                # Set the name attribute for the data
                data.name = dataset_name
                
                # Analyze the question and generate code
                code, explanation = analyze_question(question, data)
                
                return create_response(
                    {"explanation": explanation},
                    code=code,
                    message=f"Analysis for {dataset_name}: {question}"
                )
            
            except Exception as e:
                return create_response(
                    None, status="error", error=f"Error analyzing data: {str(e)}"
                )
        
        else:
            return create_response(
                None, status="error", error=f"Unknown request type: {request_type}"
            )
    
    except Exception as e:
        return create_response(None, status="error", error=f"Error processing request: {str(e)}")

def handle_notebook_request(request_str):
    """
    Handle a notebook request string.
    
    Parameters
    ----------
    request_str : str
        The JSON request string
        
    Returns
    -------
    str
        The JSON response string
    """
    try:
        request = parse_input(request_str)
        if isinstance(request, dict) and "status" in request and request["status"] == "error":
            return json.dumps(request)
        
        request_type = request.get("type")
        request_data = request.get("data", {})
        
        response = process_notebook_request(request_type, request_data)
        return json.dumps(response)
    
    except Exception as e:
        msg = f"Error processing request: {str(e)}"
        return json.dumps(create_response(None, status="error", error=msg))

if __name__ == "__main__":
    # Example usage
    test_request = json.dumps({
        "type": "get_available_datasets",
        "data": {},
        "context": {"jupyter_notebook": True}
    })
    
    response = handle_notebook_request(test_request)
    print(response)