#!/usr/bin/env python3
"""
Test file loading from CSV
"""

import pandas as pd
import pyrsm
import tempfile
import os

print("="*70)
print("FILE LOADING TEST")
print("="*70)

# Step 1: Create a test CSV file
print("\n1. CREATE TEST CSV FILE")
print("-"*70)
test_data = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D', 'E'],
    'sales': [100, 150, 200, 175, 225],
    'price': [10, 15, 20, 17.5, 22.5],
    'marketing': [5, 8, 12, 9, 13]
})

# Create temp file
temp_dir = tempfile.gettempdir()
csv_path = os.path.join(temp_dir, 'test_sales_data.csv')
test_data.to_csv(csv_path, index=False)
print(f"✓ Created: {csv_path}")
print(f"  Columns: {list(test_data.columns)}")
print(f"  Rows: {len(test_data)}")

# Step 2: Load from CSV
print("\n2. LOAD CSV FILE")
print("-"*70)
loaded_data = pd.read_csv(csv_path)
print(f"✓ Loaded from CSV")
print(f"  Shape: {loaded_data.shape}")
print("\nFirst 3 rows:")
print(loaded_data.head(3))

# Step 3: Fit regression on loaded data
print("\n3. FIT REGRESSION ON LOADED DATA")
print("-"*70)
reg = pyrsm.model.regress(loaded_data, rvar='sales', evar=['price', 'marketing'])
print("✓ Model fitted on loaded CSV data")

# Step 4: Summary
print("\n4. REGRESSION SUMMARY")
print("-"*70)
reg.summary()

# Step 5: Test different file formats
print("\n5. TEST OTHER FORMATS")
print("-"*70)

# Parquet
try:
    parquet_path = os.path.join(temp_dir, 'test_sales_data.parquet')
    test_data.to_parquet(parquet_path, index=False)
    loaded_parquet = pd.read_parquet(parquet_path)
    print(f"✓ Parquet: {parquet_path}")
    print(f"  Loaded: {loaded_parquet.shape}")
except Exception as e:
    print(f"✗ Parquet: {e}")

# JSON
try:
    json_path = os.path.join(temp_dir, 'test_sales_data.json')
    test_data.to_json(json_path, orient='records')
    loaded_json = pd.read_json(json_path)
    print(f"✓ JSON: {json_path}")
    print(f"  Loaded: {loaded_json.shape}")
except Exception as e:
    print(f"✗ JSON: {e}")

# Excel (requires openpyxl)
try:
    excel_path = os.path.join(temp_dir, 'test_sales_data.xlsx')
    test_data.to_excel(excel_path, index=False)
    loaded_excel = pd.read_excel(excel_path)
    print(f"✓ Excel: {excel_path}")
    print(f"  Loaded: {loaded_excel.shape}")
except Exception as e:
    print(f"✗ Excel: {e} (openpyxl not installed)")

print("\n" + "="*70)
print("✓ FILE LOADING TEST COMPLETE")
print("="*70)
print(f"\nTest files created in: {temp_dir}")
print("These paths can be used with data_load_file() tool")
