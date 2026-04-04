"""
User-Defined Function (UDF): L0 Norm Example

RUN COMMAND: python examples/udf_l0_norm.py

This example demonstrates how to implement a custom nonconvex penalty
(L0 norm) as a User-Defined Function (UDF) in ADMM.

Problem: Sparse signal recovery with box constraints using the L0 norm
(counts nonzero entries) as a regularizer.

Mathematical formulation:
    min_x   (1/2) ||x - y||_2^2  +  lam * ||x||_0
    s.t.    0 <= x <= 1

Where:
    x: decision variable
    y: observed noisy signal
    lam: regularization parameter
    ||x||_0: number of nonzero entries in x (nonconvex)

The proximal operator for lam * ||x||_0 is the hard-thresholding operator:
    prox(v) = v * 1_{|v| > sqrt(2*lam)}
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define the L0 Norm UDF
# ============================================================================
class L0Norm(admm.UDFBase):
    """L0 norm: counts the number of nonzero entries."""

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
# Step 2: Define problem data
# ============================================================================
y = np.array([0.2, 1.7, 0.6, 1.9])
lam = 1.0

print("=" * 70)
print("User-Defined Function (UDF): L0 Norm Example")
print("=" * 70)
print(f"Observed signal y: {y}")
print(f"Regularization (lam): {lam}")
print(f"Hard threshold: sqrt(2 * lam) = {np.sqrt(2 * lam):.6f}")
print("Box constraints: 0 <= x <= 1")
print()
print("Expected behavior: entries below threshold are zeroed,")
print("large entries are clipped to [0, 1].")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", len(y))

# ============================================================================
# Step 4: Set objective with UDF
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * L0Norm(x))

# ============================================================================
# Step 5: Add constraints
# ============================================================================
model.addConstr(x >= 0)
model.addConstr(x <= 1)
model.setOption(admm.Options.admm_max_iteration, 10000)

# ============================================================================
# Step 6: Solve and print results
# ============================================================================
model.optimize()

print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status: {model.StatusString}")
    print(f"  Optimal objective value: {model.ObjVal:.6f}")
    print(f"  Solver time: {model.SolverTime:.4f} seconds")
    print()

    x_val = np.asarray(x.X)

    print("Solution analysis:")
    print(f"  {'Index':<8} {'y_i':<10} {'x_i':<12} {'Zeroed?'}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*7}")
    for i in range(len(y)):
        zeroed = "yes" if abs(x_val[i]) < 1e-4 else "no"
        print(f"  {i:<8} {y[i]:<10.4f} {x_val[i]:<12.6f} {zeroed}")

    print()
    print(f"  Nonzero entries: {np.sum(np.abs(x_val) > 1e-4)}/{len(y)}")
    print(f"  ||x||_0 = {np.count_nonzero(np.abs(x_val) > 1e-4)}")

    # Constraint verification
    print(f"\n  Box constraint: x in [{x_val.min():.6f}, {x_val.max():.6f}]")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
