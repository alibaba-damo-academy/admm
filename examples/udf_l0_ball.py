"""
User-Defined Function (UDF): L0 Ball Indicator Example

RUN COMMAND: python examples/udf_l0_ball.py

This example demonstrates a cardinality-constrained nearest-point problem
using the L0 ball indicator as a User-Defined Function (UDF).

Problem: Find the closest point to y that has at most k nonzero entries
(cardinality constraint).

Mathematical formulation:
    min_x   (1/2) ||x - y||_2^2  +  delta_{||x||_0 <= k}(x)

Where:
    x: decision variable
    y: target vector
    k: maximum number of nonzero entries
    delta_{||x||_0 <= k}: indicator function (0 if feasible, +inf otherwise)

The proximal operator projects onto the L0 ball by keeping the k entries
with largest absolute values.
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define the L0 Ball Indicator UDF
# ============================================================================
class L0BallIndicator(admm.UDFBase):
    """L0 ball indicator: enforces at most k nonzero entries."""

    def __init__(self, arg, k=2):
        self.arg = arg
        self.k = k

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return 0.0 if np.count_nonzero(np.abs(x) > 1e-12) <= self.k else float("inf")

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        prox = np.zeros_like(v)
        keep_count = min(max(self.k, 0), v.size)
        if keep_count > 0:
            keep_idx = np.argpartition(np.abs(v), -keep_count)[-keep_count:]
            prox[keep_idx] = v[keep_idx]
        return [prox.tolist()]


# ============================================================================
# Step 2: Define problem data
# ============================================================================
y = np.array([0.2, -1.5, 0.7, 3.0])
k = 2   # maximum cardinality

print("=" * 70)
print("User-Defined Function (UDF): L0 Ball Indicator Example")
print("=" * 70)
print(f"Target vector y: {y}")
print(f"Max nonzero entries (k): {k}")
print()
print(f"Expected: keep the {k} entries with largest magnitude,")
print("zero out the rest. Should keep y[1]=-1.5 and y[3]=3.0.")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", len(y))

# ============================================================================
# Step 4: Set objective with UDF
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + L0BallIndicator(x, k=k))

# ============================================================================
# Step 5: Add constraints
# ============================================================================
# No additional constraints

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
    print(f"  {'Index':<8} {'y_i':<10} {'x_i':<12} {'Kept?'}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*5}")
    for i in range(len(y)):
        kept = "yes" if abs(x_val[i]) > 1e-4 else "no"
        print(f"  {i:<8} {y[i]:<10.4f} {x_val[i]:<12.6f} {kept}")

    print()
    print(f"  Nonzero entries: {np.sum(np.abs(x_val) > 1e-4)} (limit: {k})")
    print(f"  ||x||_0 = {np.count_nonzero(np.abs(x_val) > 1e-4)}")

    # Approximation error
    approx_error = np.linalg.norm(x_val - y)
    print(f"\n  Approximation error (||x - y||_2): {approx_error:.6f}")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
