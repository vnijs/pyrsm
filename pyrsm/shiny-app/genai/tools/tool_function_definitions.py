# JSON representation of the single_mean class
single_mean_tool = {
    "name": "single_mean",
    "description": "Perform single-mean hypothesis testing.",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": "The name of the dataframe available in the environment to be used for hypothesis testing.",
            },
            "var": {"type": "string", "description": "The variable/column name to test."},
            "alt_hyp": {
                "type": "string",
                "enum": ["two-sided", "greater", "less"],
                "description": "The alternative hypothesis.",
            },
            "conf": {"type": "number", "description": "The confidence level for the test.", "default": 0.95},
            "comp_value": {"type": "number", "description": "The comparison value for the test.", "default": 0},
            "dec": {"type": "number", "description": "The number of decimals to show in the summary output", "default": 3},
            "plots": {"type": "string", "enum": ["hist", "sim"], "description": "Plots the results of the hypothesis test. If the 'hist' plot is selected a histogram of the numeric variable will be shown. The solid black line in the histogram shows the sample mean. The dashed black lines show the confidence interval around the sample mean. The solid red line shows the comparison value (i.e., the value under the null-hypothesis). If the red line does not fall within the confidence interval we can reject the null-hypothesis in favor of the alternative at the specified confidence level (e.g., 0.95). The 'sim' plot is not currently available.", "default": "hist"},
        },
        "required": ["data", "var"],
    },
}

# JSON representation of the compare_means class
compare_means_tool = {
    "name": "compare_means",
    "description": "Perform comparison of means hypothesis testing.",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": "The name of the dataframe available in the environment to be used for hypothesis testing.",
            },
            "var1": {
                "type": "string",
                "description": "The first variable/column name to test. This can be numeric or categorical.",
            },
            "var2": {"type": "string", "description": "The second variable/column name to test. This must be numeric."},
            "comb": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of string tuples representing combinations of variables for comparison. Defaults to all possible combinations.",
                "default": [],
            },
            "alt_hyp": {
                "type": "string",
                "enum": ["two-sided", "greater", "less"],
                "description": "The alternative hypothesis.",
                "default": "two-sided",
            },
            "conf": {"type": "number", "description": "The confidence level for the test.", "default": 0.95},
            "sample_type": {
                "type": "string",
                "enum": ["independent", "paired"],
                "description": "Type of samples ('independent' or 'paired').",
                "default": "independent",
            },
            "adjust": {
                "type": "string",
                "enum": [None, "bonferroni"],
                "description": "Adjustment for multiple testing (e.g., None or 'bonferroni').",
                "default": None,
            },
            "test_type": {
                "type": "string",
                "enum": ["t-test", "wilcox"],
                "description": "The type of test to use ('t-test' or 'wilcox').",
                "default": "t-test",
            },
            "dec": {
                "type": "integer",
                "default": 3,
                "description": "The number of decimal places to display in the summary.",
            },
            "extra": {
                "type": "boolean",
                "default": False,
                "description": "Whether to include extra details in the summary output.",
            },
            "plots": {
                "type": "string",
                "enum": ["scatter", "box", "density", "bar"],
                "description": "The type of plot to create ('scatter', 'box', 'density', 'bar').",
                "default": "scatter",
            },
            "nobs": {
                "type": "integer",
                "description": "The number of observations to plot. Defaults to all available datapoints.",
                "default": None,
            },
        },
        "required": ["data", "var1", "var2"],
    },
}

linear_regression_tool = {
    "name": "regress",
    "description": "Perform linear regression modeling.",
    "parameters": {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": "The name of the dataframe available in the environment to be used for linear regression analysis.",
            },
            "rvar": {
                "type": "string",
                "description": "The response or dependent variable to be used in the regression model. This must be a numeric variable. Ask the user to select this variable from a list of numeric variables.",
            },
            "evar": {
                "type": "string",
                "description": "A required list of explanatory or independent variables to be used in the regression model as a list of the variable names. If therer fewer than 10 explanatory variables in the data use all of them to start. Else, ask the user to select these variables from a list of all available variables but exclude the response variable.",
            },
            "ivar": {
                "type": "string",
                "description": "An optional list of interaction terms to included as explanatory (independent) variable in the regression model. Ask the user if they want to include interactions terms. If they do, ask them select the interactions as a pair of explanatory variables. For example, if we have variables x1 and x2 in the model as explanatory variables in the model, a possible interaction term would be included as 'x1:x2'. Notice the colon between the variable names. Do NOT include interactions unless the user explicity asks for them.",
            },
        },
        "required": ["data", "rvar", "evar"],
    },
}

tools = [
    {
        "type": "function",
        "function": single_mean_tool,
    },
    {
        "type": "function",
        "function": compare_means_tool,
    },
    {
        "type": "function",
        "function": linear_regression_tool,
    },

]
