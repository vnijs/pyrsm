#!/usr/bin/env python3
"""
Test data loading workflow
"""

import pyrsm

print("="*70)
print("DATA LOADING WORKFLOW TEST")
print("="*70)

# Step 1: List available datasets
print("\n1. LIST AVAILABLE DATASETS (model package)")
print("-"*70)
data_dict, desc_dict = pyrsm.load_data(pkg="model")
print(f"Found {len(data_dict)} datasets:")
for name in sorted(list(data_dict.keys())[:5]):
    print(f"  • {name}")
print("  ...")

# Step 2: Load specific dataset
print("\n2. LOAD SPECIFIC DATASET")
print("-"*70)
data, description = pyrsm.load_data(pkg="model", name="diamonds")
print(f"✓ Loaded: diamonds")
print(f"  Shape: {data.shape}")
print(f"  Columns: {list(data.columns)}")

# Step 3: Dataset info
print("\n3. DATASET INFO")
print("-"*70)
print(f"First 3 rows:")
print(data.head(3))
print(f"\nData types:")
print(data.dtypes)

# Step 4: Fit regression on loaded data
print("\n4. FIT REGRESSION ON LOADED DATA")
print("-"*70)
reg = pyrsm.model.regress(data, rvar='price', evar=['carat', 'depth', 'table'])
print("✓ Model fitted on diamonds dataset")

# Step 5: Get summary
print("\n5. SUMMARY")
print("-"*70)
reg.summary(dec=2)

print("\n" + "="*70)
print("✓ DATA LOADING WORKFLOW COMPLETE")
print("="*70)
print("\nKey insight: Load data once, use multiple times")
print("No need to reload for each analysis!")
