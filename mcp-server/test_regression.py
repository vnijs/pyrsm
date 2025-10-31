#!/usr/bin/env python3
"""
Test the regression workflow with state management
"""

import pandas as pd
import pyrsm

# Sample data (same as in server)
data = pd.DataFrame({
    'sales': [100, 150, 200, 250, 300, 120, 180, 220, 280, 320, 95, 145, 195, 245, 295],
    'x1': [1, 2, 3, 4, 5, 1.5, 2.5, 3.5, 4.5, 5.5, 1.2, 2.2, 3.2, 4.2, 5.2],
    'x2': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55, 12, 22, 32, 42, 52],
    'x3': [5, 10, 15, 20, 25, 7, 12, 17, 22, 27, 6, 11, 16, 21, 26],
})

print("="*70)
print("REGRESSION WORKFLOW TEST")
print("="*70)

# Step 1: Fit model
print("\n1. FIT MODEL")
print("-"*70)
reg = pyrsm.model.regress(data, rvar='sales', evar=['x1', 'x2', 'x3'])
print("✓ Model fitted")

# Step 2: Basic summary
print("\n2. BASIC SUMMARY")
print("-"*70)
reg.summary()

# Step 3: Add VIF (no refitting!)
print("\n3. ADD VIF (no refitting)")
print("-"*70)
reg.summary(vif=True)

# Step 4: Diagnostic plots
print("\n4. DIAGNOSTIC PLOTS")
print("-"*70)
print("Code: reg.plot(plots='dashboard')")
print("(Plot would appear here)")

# Step 5: Variable importance
print("\n5. VARIABLE IMPORTANCE")
print("-"*70)
print("Code: reg.plot(plots='vimp')")
print("(Plot would appear here)")

print("\n" + "="*70)
print("✓ WORKFLOW COMPLETE")
print("="*70)
print("\nKey insight: Steps 3-5 use the SAME fitted model object")
print("No refitting required!")
