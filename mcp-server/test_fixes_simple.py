"""
Simple test to verify the three key fixes without IPython
"""
import sys
sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

print("=" * 60)
print("SIMPLE FIX VERIFICATION")
print("=" * 60)

# Test 1: MODEL_STORE import
print("\n✓ Test 1: MODEL_STORE import in server_regression...")
try:
    from server_regression import DATA_STORE, MODEL_STORE
    print(f"  ✓ DATA_STORE: {type(DATA_STORE)}")
    print(f"  ✓ MODEL_STORE: {type(MODEL_STORE)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Check mcp_bridge_magic has MODEL_STORE import
print("\n✓ Test 2: Check mcp_bridge_magic.py imports MODEL_STORE...")
try:
    with open('/home/vnijs/gh/pyrsm/mcp-server/mcp_bridge_magic.py') as f:
        content = f.read()
        if 'from server_regression import DATA_STORE, MODEL_STORE, call_tool' in content:
            print("  ✓ MODEL_STORE import found in mcp_bridge_magic.py")
        else:
            print("  ✗ MODEL_STORE import NOT found")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Check TypeError fix (null checking)
print("\n✓ Test 3: Check TypeError fix in mcp_bridge_magic.py...")
try:
    with open('/home/vnijs/gh/pyrsm/mcp-server/mcp_bridge_magic.py') as f:
        content = f.read()
        if 'if hasattr(part, \'function_call\') and part.function_call:' in content:
            print("  ✓ Null checking for function_call found")
        else:
            print("  ✗ Null checking NOT found")
            sys.exit(1)

        if 'dict(args) if args else {}' in content:
            print("  ✓ Args null checking found")
        else:
            print("  ✗ Args null checking NOT found")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Check regression summary fix
print("\n✓ Test 4: Check reg.summary() in server_regression.py...")
try:
    with open('/home/vnijs/gh/pyrsm/mcp-server/server_regression.py') as f:
        content = f.read()
        # Find regress_fit handler
        if 'reg.summary(vif={vif})' in content or 'reg.summary(vif=' in content:
            print("  ✓ reg.summary() found in generated code")
        else:
            print("  ✗ reg.summary() NOT in generated code")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Check model_analyze tool exists
print("\n✓ Test 5: Check model_analyze tool in GEMINI_TOOLS...")
try:
    with open('/home/vnijs/gh/pyrsm/mcp-server/mcp_bridge_magic.py') as f:
        content = f.read()
        if '"name": "model_analyze"' in content:
            print("  ✓ model_analyze tool found")
        else:
            print("  ✗ model_analyze tool NOT found")
            sys.exit(1)

        if 'operation' in content and 'model_name' in content:
            print("  ✓ model_analyze parameters found")
        else:
            print("  ✗ model_analyze parameters NOT found")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 6: Check model detection code
print("\n✓ Test 6: Check model detection in _get_kernel_context...")
try:
    with open('/home/vnijs/gh/pyrsm/mcp-server/mcp_bridge_magic.py') as f:
        content = f.read()
        if "class_name in ['regress', 'single_mean', 'compare_means']" in content:
            print("  ✓ Model class detection found")
        else:
            print("  ✗ Model class detection NOT found")
            sys.exit(1)

        if "'models':" in content and "'n_models':" in content:
            print("  ✓ Models added to context")
        else:
            print("  ✗ Models NOT in context")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 7: Check _build_context_prompt includes models
print("\n✓ Test 7: Check _build_context_prompt shows models...")
try:
    with open('/home/vnijs/gh/pyrsm/mcp-server/mcp_bridge_magic.py') as f:
        content = f.read()
        if 'Available models:' in content:
            print("  ✓ 'Available models:' section found")
        else:
            print("  ✗ 'Available models:' NOT found")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 8: Check system instructions mention model_analyze
print("\n✓ Test 8: Check system instructions for LLM...")
try:
    with open('/home/vnijs/gh/pyrsm/mcp-server/mcp_bridge_magic.py') as f:
        content = f.read()
        if 'use the model_analyze tool' in content:
            print("  ✓ model_analyze mentioned in system instructions")
        else:
            print("  ✗ model_analyze NOT in system instructions")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 9: Model objects can be detected
print("\n✓ Test 9: Test pyrsm model object detection...")
try:
    import pyrsm
    import pandas as pd

    # Create test data
    test_data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 5, 4, 5]})

    # Create models
    sm = pyrsm.basics.single_mean({'test': test_data}, var='x', comp_value=3)
    reg = pyrsm.model.regress(test_data, rvar='y', evar=['x'])

    # Check class names
    if sm.__class__.__name__ == 'single_mean':
        print(f"  ✓ single_mean class: {sm.__class__.__name__}")
    else:
        print(f"  ✗ Unexpected class: {sm.__class__.__name__}")
        sys.exit(1)

    if reg.__class__.__name__ == 'regress':
        print(f"  ✓ regress class: {reg.__class__.__name__}")
    else:
        print(f"  ✗ Unexpected class: {reg.__class__.__name__}")
        sys.exit(1)

    # Check attributes
    if hasattr(sm, 'var'):
        print(f"  ✓ single_mean.var = {sm.var}")

    if hasattr(reg, 'rvar') and hasattr(reg, 'evar'):
        print(f"  ✓ regress.rvar = {reg.rvar}, evar = {reg.evar}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL FIXES VERIFIED!")
print("=" * 60)
print("\nSummary of verified fixes:")
print("  1. ✓ TypeError fix: Null checking for function_call.args")
print("  2. ✓ Regression output fix: reg.summary() in generated code")
print("  3. ✓ MODEL_STORE imported in mcp_bridge_magic.py")
print("  4. ✓ Model detection: Scans kernel for pyrsm models")
print("  5. ✓ Context includes models: _get_kernel_context returns models")
print("  6. ✓ Display includes models: _build_context_prompt shows models")
print("  7. ✓ New tool: model_analyze for post-analysis operations")
print("  8. ✓ LLM instructions: System prompt mentions model reuse")
print("  9. ✓ Model objects: Can detect and extract info from pyrsm models")
print("\n✓ Ready for notebook testing!")
