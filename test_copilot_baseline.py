"""
Copilot Baseline Test
=====================

Instructions:
1. Open this file in VS Code with GitHub Copilot enabled
2. Put your cursor at the end of each comment
3. Press Tab to accept Copilot's suggestion (or wait for it)
4. Record what Copilot suggests (pyrsm vs generic pandas/sklearn)

Test Results: Record for each test below
- ✓ = Suggests pyrsm
- ✗ = Suggests generic libraries (pandas, sklearn, statsmodels, etc.)
- ? = Suggests nothing useful
"""

import pandas as pd

# Load sample data
data = pd.DataFrame({
    'sales': [100, 150, 200, 250, 300],
    'x1': [1, 2, 3, 4, 5],
    'x2': [2, 4, 6, 8, 10],
    'x3': [3, 6, 9, 12, 15],
    'price': [95, 105, 98, 102, 110],
    'treatment': ['A', 'B', 'A', 'B', 'A'],
    'score': [85, 90, 88, 92, 87]
})

# TEST 1: I want to run a linear regression with 'sales' as the response and x1, x2, x3 as the explanatory variables. Show me the summary output


# TEST 2: Test if the mean of 'price' is significantly different from 100


# TEST 3: Compare means between two groups 'treatment' for variable 'score'


# TEST 4: Load the diamonds dataset from pyrsm


# TEST 5: Test if proportion of values in 'treatment' equal to 'A' is different from 0.5


# TEST 6: Run a correlation analysis between x1, x2, and x3


# TEST 7: Create a logistic regression (assume we have binary target column)


# TEST 8: Calculate confidence intervals for my regression coefficients


# TEST 9: Create a random forest model for classification


# TEST 10: Plot residuals from a linear regression model
