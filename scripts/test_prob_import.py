#!/usr/bin/env python
"""Smoke test for pyrsm.basics.prob_calc import and behavior.

Run this with: uv run python scripts/test_prob_import.py
"""
import sys

def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)

def ok(msg):
    print("OK:", msg)


def main():
    try:
        import pyrsm as rsm
    except Exception as e:
        fail(f"import pyrsm failed: {e!r}")

    try:
        prob_mod = rsm.basics.prob_calc
    except Exception as e:
        fail(f"accessing rsm.basics.prob_calc failed: {e!r}")

    ok(f"found rsm.basics.prob_calc: {prob_mod}")

    dists = ["binom", "pois", "norm"]
    for d in dists:
        name = f"prob_{d}"
        has = hasattr(prob_mod, name)
        print(f"has {name}: {has}")

    # Try dispatcher call for binom
    try:
        pc = rsm.basics.prob_calc("binom", n=5, p=0.4, lb=0, ub=5)
    except Exception as e:
        fail(f"dispatcher call failed: {e!r}")

    ok(f"dispatcher returned object: {type(pc)}")

    # Check summary
    if not hasattr(pc, "summary"):
        fail("returned object has no 'summary' method")
    try:
        out = pc.summary(dec=3)
        print("summary output (truncated):\n", str(out)[:400])
    except Exception as e:
        fail(f"calling pc.summary failed: {e!r}")

    # Check plot method exists (don't render)
    if not hasattr(pc, "plot"):
        fail("returned object has no 'plot' method")

    ok("plot method exists (not rendering) and summary ran OK")

    print("All smoke tests passed")


if __name__ == "__main__":
    main()
