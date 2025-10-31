#!/usr/bin/env python3
"""
Test prompt_magic in IPython environment
"""

# Start IPython embedded session
from IPython import embed
from IPython.terminal.embed import InteractiveShellEmbed
import sys

sys.path.insert(0, '/home/vnijs/gh/pyrsm/mcp-server')

# Create IPython shell
ipython = InteractiveShellEmbed()

print("="*70)
print("PROMPT MAGIC TEST - Interactive IPython Session")
print("="*70)
print()
print("Try these commands:")
print()
print("  %load_ext prompt_magic")
print()
print("  import pyrsm")
print("  diamonds, desc = pyrsm.load_data('diamonds', 'model')")
print()
print("  %%prompt")
print("  Fit a regression with price as response and carat, depth as predictors")
print()
print("Type 'exit()' to quit")
print("="*70)
print()

# Start interactive session
ipython()
