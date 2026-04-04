"""
UDF Introduction: L0 Norm (User Guide Walkthrough)

RUN COMMAND: python examples/udf_intro_l0.py

This is the introductory UDF example from the User Guide, demonstrating how
to define a custom proximal operator for the L0 norm.

Mathematical formulation:
    min_x   (1/2) ||x - y||_2^2 + lam * ||x||_0

Where:
    ||x||_0 = number of nonzero entries (nonconvex)
    Proximal operator: hard thresholding at sqrt(2 * lam)

Data: y = [0.2, 2.0, 0.6, 2.2], lam = 1.0
Expected: x* ~ [0, 2, 0, 2.2] (entries below threshold removed)
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF
# ============================================================================


class L0Norm(admm.UDFBase):
    """L0 norm: count of nonzero entries."""

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.count_nonzero(np.abs(x) > 1e-12))

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        threshold = np.sqrt(2.0 * lamb)
        prox = np.where(np.abs(v) <= threshold, 0.0, v)
        return [prox.tolist()]


# ============================================================================
# Step 2: Set up data
# ============================================================================
y = np.array([0.2, 2.0, 0.6, 2.2])
lam = 1.0

print("=" * 70)
print("UDF Introduction: L0 Norm (User Guide Walkthrough)")
print("=" * 70)
print(f"y = {y}")
print(f"lam = {lam}")
print(f"Hard threshold = sqrt(2*lam) = {np.sqrt(2*lam):.4f}")
print()

# ============================================================================
# Step 3: Create model and variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", len(y))

# ============================================================================
# Step 4: Set objective (no constraints — unconstrained L0 problem)
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * L0Norm(x))

# ============================================================================
# Step 5: Solve
# ============================================================================
model.optimize()

# ============================================================================
# Step 6: Results
# ============================================================================
print()
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    x_val = np.asarray(x.X)
    print(f"  Status: {model.StatusString}")
    print(f"  Objective value: {model.ObjVal:.6f}")       # Expected: ~ 2.2
    print(f"  x = {x_val}")                                # Expected: ~ [0, 2, 0, 2.2]
    print(f"  Nonzeros: {np.count_nonzero(np.abs(x_val) > 1e-6)}")
else:
    print(f"  Optimization failed: {model.StatusString}")

print("=" * 70)
