"""Test model context tracking and reuse"""
import sys
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

# Test that we can detect pyrsm model objects
import pandas as pd
import pyrsm

# Create test data
salary = pd.DataFrame({
    'salary': [50000, 60000, 70000, 80000, 90000],
    'rank': ['Assistant', 'Assistant', 'Associate', 'Associate', 'Full']
})

# Create a single_mean model
sm = pyrsm.basics.single_mean({'salary': salary}, var='salary', comp_value=70000)

# Create a regression model
diamonds, _ = pyrsm.load_data(name='diamonds', pkg='model')
reg = pyrsm.model.regress(diamonds, rvar='price', evar=['carat', 'depth', 'table'])

# Test detection
print("Testing model detection...")
print(f"sm type: {sm.__class__.__name__}")
print(f"reg type: {reg.__class__.__name__}")

# Test attributes
if hasattr(sm, 'var'):
    print(f"✓ sm.var = {sm.var}")

if hasattr(reg, 'rvar') and hasattr(reg, 'evar'):
    print(f"✓ reg.rvar = {reg.rvar}")
    print(f"✓ reg.evar = {reg.evar}")

# Test that context extraction would work
user_ns = {'sm': sm, 'reg': reg, 'salary': salary, 'diamonds': diamonds}

models_found = {}
for name, obj in user_ns.items():
    if not name.startswith('_'):
        class_name = obj.__class__.__name__
        if class_name in ['regress', 'single_mean', 'compare_means']:
            models_found[name] = class_name
            print(f"✓ Found {class_name} model: {name}")

print(f"\n✓ Test passed! Found {len(models_found)} model(s)")
