# PYRSM MCP Server Examples

This document provides example prompts and responses for using the PYRSM MCP server with GitHub Copilot Chat in VS Code.

## Dataset Management Examples

### Example 1: Listing Available Datasets

**Prompt:**
```
Which datasets are available?
```

**Expected Interaction:**
Copilot Chat will use the `get_available_datasets` tool to list all available datasets in the PYRSM package, showing their names, dimensions, and short descriptions.

### Example 2: Loading a Specific Dataset

**Prompt:**
```
Load the demand_uk dataset
```

**Expected Interaction:**
Copilot Chat will use the `load_dataset` tool to load the demand_uk dataset and provide information about it, including its shape, first few rows, and column names.

## Data Analysis Examples

### Example 3: Analyzing Data with a Question

**Prompt:**
```
Is the demand column larger than 1750?
```

**Expected Interaction:**
Copilot Chat will use the `analyze_data` tool to generate and run code that checks if values in the "demand" column are greater than 1750, providing both numerical results and a visualization.

### Example 4: Correlation Analysis

**Prompt:**
```
What's the correlation between price and demand?
```

**Expected Interaction:**
Copilot Chat will analyze the correlation between the specified columns, generating code that computes correlation coefficients and creates visualization of the relationship.

## Statistical Testing Examples

### Example 5: Generating Sample Data

**Prompt:**
```
Generate a sample dataset with 100 observations, a mean of 75, and a standard deviation of 10.
```

**Expected Interaction:**
Copilot Chat will use the `generate_sample_data` tool to create a dataset with the specified parameters and report summary statistics.

### Example 6: Performing a Simple Hypothesis Test

**Prompt:**
```
Perform a single mean test on a dataset to determine if the mean is greater than 70.
```

**Expected Interaction:**
Copilot Chat will use the `single_mean_test` tool with the alternative hypothesis set to "greater" and a comparison value of 70, then display the results including the test statistic, p-value, and confidence interval.

### Example 7: Interpreting Results

**Prompt:**
```
I have a single mean test with t-value 2.34, p-value 0.021, and confidence interval [72.1, 78.3]. What does this mean?
```

**Expected Interaction:**
Copilot Chat will explain the statistical significance, what the confidence interval represents, and whether the null hypothesis can be rejected.

## Jupyter Notebook Integration Examples

### Example 8: Listing Datasets in a Jupyter Notebook

**Prompt:**
```
Which datasets are available? I want to use one in my notebook.
```

**Expected Interaction:**
Copilot Chat will detect that this is a Jupyter notebook context and will generate Python code to list available datasets, then insert the code into a notebook cell for execution.

### Example 9: Loading a Dataset in a Notebook

**Prompt:**
```
Load the demand_uk dataset into my notebook
```

**Expected Interaction:**
Copilot Chat will generate Python code to load the dataset using pyrsm's load_data function, insert it into a notebook cell, and run it to show the dataset information.

### Example 10: Analyzing Data in a Notebook

**Prompt:**
```
I'm using the demand_uk dataset. Is demand larger than 1750? Please create a visualization.
```

**Expected Interaction:**
Copilot Chat will generate Python code that analyzes the demand column, checks values against the threshold, and creates a histogram with appropriate annotations. This code will be inserted into a notebook cell and executed.

---

**Note:** These examples assume the MCP server is properly configured and running in VS Code. The actual behavior may vary depending on the version of GitHub Copilot Chat and how it interacts with MCP tools.