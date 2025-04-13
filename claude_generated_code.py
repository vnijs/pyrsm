# Generated code for: Is demand greater than 1750?
# Dataset: demand_uk

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyrsm as rsm

# Load the dataset
demand_uk, demand_uk_description = rsm.load_data(name='demand_uk')


# Check if values in 'demand' are greater than 1750.0
import matplotlib.pyplot as plt

# Calculate basic statistics
column_data = demand_uk['demand']
mean_value = column_data.mean()
greater_than_count = (column_data > 1750.0).sum()
percentage = (greater_than_count / len(column_data)) * 100

# Display results
print(f"Mean of 'demand': {mean_value:.2f}")
print(f"Number of values > 1750.0: {greater_than_count} of {len(column_data)}")
print(f"Percentage greater than 1750.0: {percentage:.2f}%")

# Create a histogram to visualize the distribution
plt.figure(figsize=(10, 6))
plt.hist(column_data, bins=20, alpha=0.7, color='skyblue')
plt.axvline(x=1750.0, color='red', linestyle='--', 
           label=f'Threshold: 1750.0')
plt.axvline(x=mean_value, color='green', linestyle='-', 
           label=f'Mean: {mean_value:.2f}')
plt.title(f"Distribution of 'demand'")
plt.xlabel(f'demand')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
