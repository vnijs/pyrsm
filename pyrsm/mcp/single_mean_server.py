"""
MCP server for the single_mean module of pyrsm.
This server provides functionality to perform single-mean hypothesis testing.
"""

import json
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from pyrsm.basics.single_mean import single_mean


def create_response(obj, message=None, status="success", error=None):
    """Create a standardized response format."""
    response = {"status": status}
    
    if message:
        response["message"] = message
    
    if error:
        response["error"] = str(error)
        
    if obj is not None:
        response["data"] = obj
        
    return response


def parse_input(data_str):
    """Parse the input JSON data."""
    try:
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError as e:
        return create_response(None, status="error", error=f"Invalid JSON: {str(e)}")


def create_sample_data():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)
    data = pd.DataFrame({
        "values": np.random.normal(100, 15, 50)
    })
    return data


def perform_single_mean_test(data, var, alt_hyp="two-sided", conf=0.95, comp_value=0):
    """
    Perform a single mean test.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input data for the hypothesis test
    var : str
        The variable/column name to test
    alt_hyp : str, optional
        The alternative hypothesis ('two-sided', 'greater', 'less')
    conf : float, optional
        The confidence level for the test
    comp_value : float, optional
        The comparison value for the test
        
    Returns
    -------
    dict
        The test results
    """
    try:
        # Create a single_mean object
        sm_test = single_mean(data, var, alt_hyp, conf, comp_value)
        
        # Capture text output from summary method
        buffer = io.StringIO()
        import sys
        original_stdout = sys.stdout
        sys.stdout = buffer
        sm_test.summary()
        sys.stdout = original_stdout
        summary_text = buffer.getvalue()
        
        # Generate plot
        plt.figure(figsize=(10, 6))
        sm_test.plot()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()
        
        # Prepare response
        result = {
            "mean": float(sm_test.mean),
            "t_value": float(sm_test.t_val),
            "p_value": float(sm_test.p_val),
            "confidence_interval": [float(sm_test.ci[0]), float(sm_test.ci[1])],
            "standard_deviation": float(sm_test.sd),
            "standard_error": float(sm_test.se),
            "sample_size": int(sm_test.n),
            "missing_values": int(sm_test.n_missing),
            "summary_text": summary_text,
            "plot": img_str
        }
        
        return create_response(result, message="Single mean test completed successfully")
    
    except Exception as e:
        return create_response(None, status="error", error=str(e))


def process_request(request_type, request_data):
    """
    Process a request based on its type.
    
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
    if request_type == "sample_data":
        data = create_sample_data()
        return create_response(data.to_dict())
    
    elif request_type == "test":
        # Parse and validate input parameters
        if not isinstance(request_data, dict):
            return create_response(None, status="error", error="Invalid request data format")
        
        # Use sample data if no data provided
        data = request_data.get("data")
        if data is None:
            df = create_sample_data()
        else:
            try:
                df = pd.DataFrame(data)
            except Exception as e:
                return create_response(None, status="error", error=f"Error creating DataFrame: {str(e)}")
        
        var = request_data.get("var")
        if not var:
            if "values" in df.columns:
                var = "values"
            else:
                var = df.columns[0]
        
        alt_hyp = request_data.get("alt_hyp", "two-sided")
        conf = float(request_data.get("conf", 0.95))
        comp_value = float(request_data.get("comp_value", 0))
        
        return perform_single_mean_test(df, var, alt_hyp, conf, comp_value)
    
    else:
        return create_response(None, status="error", error=f"Unknown request type: {request_type}")


def handle_request(request_str):
    """
    Handle a request string.
    
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
        
        response = process_request(request_type, request_data)
        return json.dumps(response)
    
    except Exception as e:
        return json.dumps(create_response(None, status="error", error=f"Error processing request: {str(e)}"))


if __name__ == "__main__":
    # Example usage
    test_request = json.dumps({
        "type": "test",
        "data": {
            "var": "values",
            "alt_hyp": "two-sided",
            "conf": 0.95,
            "comp_value": 95
        }
    })
    
    response = handle_request(test_request)
    print(response)