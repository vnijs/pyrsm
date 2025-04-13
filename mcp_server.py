"""
MCP Server for pyrsm package.
This server provides tools to perform statistical analysis using pyrsm.

To run:
    mcp dev mcp_server.py
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Literal, Any

from mcp.server.fastmcp import FastMCP

from pyrsm.basics.single_mean import single_mean
from pyrsm.example_data import load_data

# Initialize MCP server
mcp = FastMCP("pyrsm")

# Dictionary to store loaded datasets in memory across calls
session_data = {}

@mcp.tool()
def get_available_datasets() -> Dict:
    """
    List all available datasets in the pyrsm package.
    
    Returns:
        Dictionary containing information about the available datasets
    """
    # Get all datasets
    all_data, all_descriptions = load_data()
    
    # Format the result for display
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
    
    return {
        "datasets": datasets,
        "count": len(datasets)
    }

@mcp.tool()
def load_dataset(dataset_name: str) -> Dict:
    """
    Load a specific dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Dictionary containing the dataset information and sample data
    """
    # Try to find which package the dataset is in
    data = None
    description = None
    
    for pkg in ["data", "design", "basics", "model", "multivariate"]:
        try:
            data, description = load_data(pkg=pkg, name=dataset_name)
            break
        except:
            continue
    
    # If we can't find it by package, try loading it directly
    if data is None:
        try:
            data, description = load_data(name=dataset_name)
        except:
            raise ValueError(f"Dataset '{dataset_name}' not found")
    
    # Store in session for future use
    session_data[dataset_name] = data
    
    # Format the result
    result = {
        "name": dataset_name,
        "rows": len(data),
        "columns": len(data.columns),
        "column_names": list(data.columns),
        "description": description,
        "head": data.head().to_dict(),
        "code_sample": f"import pyrsm as rsm\n{dataset_name}, {dataset_name}_description = rsm.load_data(name='{dataset_name}')"
    }
    
    return result

@mcp.tool()
def analyze_data(dataset_name: str, question: str) -> Dict:
    """
    Analyze a dataset based on a natural language question.
    
    Args:
        dataset_name: Name of the dataset to analyze
        question: The data analysis question to answer
        
    Returns:
        Dictionary containing the analysis results and generated code
    """
    # Load the dataset if it's not already in session
    if dataset_name not in session_data:
        load_dataset(dataset_name)
    
    data = session_data[dataset_name]
    
    # Analyze the question and generate code
    from pyrsm.mcp.notebook_server import analyze_question
    
    # Set the name attribute for the data
    data.name = dataset_name
    
    # Generate code for the analysis
    code, explanation = analyze_question(question, data)
    
    return {
        "dataset": dataset_name,
        "question": question,
        "explanation": explanation,
        "generated_code": code
    }

@mcp.tool()
def generate_sample_data(n: int = 50, mean: float = 100, std: float = 15, seed: int = 42) -> Dict:
    """
    Generate sample data for statistical analysis.
    
    Args:
        n: Number of samples to generate
        mean: Mean value of the normal distribution
        std: Standard deviation of the normal distribution
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing the generated data
    """
    np.random.seed(seed)
    data = pd.DataFrame({
        "values": np.random.normal(mean, std, n)
    })
    
    # Store in session
    dataset_name = f"sample_data_{seed}"
    session_data[dataset_name] = data
    
    return {
        "name": dataset_name,
        "data": data.to_dict(),
        "summary": {
            "mean": float(data["values"].mean()),
            "std": float(data["values"].std()),
            "min": float(data["values"].min()),
            "max": float(data["values"].max()),
            "n": len(data),
        },
        "code_sample": f"import pandas as pd\nimport numpy as np\n\nnp.random.seed({seed})\n{dataset_name} = pd.DataFrame({{\n    \"values\": np.random.normal({mean}, {std}, {n})\n}})"
    }

@mcp.tool()
def single_mean_test(
    data: Optional[Dict] = None,
    dataset_name: Optional[str] = None,
    var: str = "values",
    alt_hyp: Literal["two-sided", "greater", "less"] = "two-sided",
    conf: float = 0.95,
    comp_value: float = 0,
    include_plot: bool = True
) -> Dict:
    """
    Perform a single mean hypothesis test.
    
    Args:
        data: Input data as a dictionary (if None, dataset_name must be provided)
        dataset_name: Name of a dataset already loaded (if data is None)
        var: Name of the variable/column to test
        alt_hyp: Alternative hypothesis ('two-sided', 'greater', 'less')
        conf: Confidence level (0-1)
        comp_value: Comparison value for the test
        include_plot: Whether to include a plot in the results
        
    Returns:
        Dictionary containing the test results
    """
    # Determine the data source
    if data is not None:
        df = pd.DataFrame(data)
    elif dataset_name is not None and dataset_name in session_data:
        df = session_data[dataset_name]
    else:
        # If no data provided, generate sample data
        sample_data = generate_sample_data()
        df = pd.DataFrame(sample_data["data"])
        dataset_name = sample_data["name"]
    
    # Set the name for code generation
    df_name = dataset_name if dataset_name else "data"
    
    # Perform the test
    sm_test = single_mean(df, var, alt_hyp, conf, comp_value)
    
    # Capture text output from summary method
    buffer = io.StringIO()
    import sys
    original_stdout = sys.stdout
    sys.stdout = buffer
    sm_test.summary()
    sys.stdout = original_stdout
    summary_text = buffer.getvalue()
    
    # Generate code for the test
    code = f"""
# Single mean hypothesis test for {var}
import pyrsm as rsm
from pyrsm.basics.single_mean import single_mean

# Perform the test
test_result = single_mean({df_name}, 
                          var='{var}', 
                          alt_hyp='{alt_hyp}', 
                          conf={conf}, 
                          comp_value={comp_value})

# Display summary
test_result.summary()

# Create visualization
test_result.plot()
"""
    
    # Create result dictionary
    result = {
        "mean": float(sm_test.mean),
        "t_value": float(sm_test.t_val),
        "p_value": float(sm_test.p_val),
        "confidence_interval": [float(sm_test.ci[0]), float(sm_test.ci[1])],
        "standard_deviation": float(sm_test.sd),
        "standard_error": float(sm_test.se),
        "sample_size": int(sm_test.n),
        "missing_values": int(sm_test.n_missing),
        "degrees_freedom": int(sm_test.df),
        "test_result": "Reject H0" if sm_test.p_val < (1 - conf) else "Fail to reject H0",
        "summary_text": summary_text,
        "generated_code": code
    }
    
    # Generate and include plot if requested
    if include_plot:
        plt.figure(figsize=(10, 6))
        sm_test.plot()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()
        
        result["plot"] = img_str
    
    return result

if __name__ == "__main__":
    mcp.run()