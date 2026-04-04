"""
UDF Simplex Indicator Example

RUN COMMAND: python examples/udf_simplex.py

This example demonstrates projection onto the probability simplex via a
User-Defined Function (UDF). The indicator enforces x >= 0 and sum(x) = r,
and its proximal operator uses a sorting-based Euclidean projection.

Problem: Find the closest point x on the simplex to a given vector y.

Mathematical formulation:
    min_x   (1/2) ||x - y||_2^2  +  delta_{Delta_r}(x)

Where:
    Delta_r = {x : x >= 0, sum(x) = r}  (probability simplex scaled by r)
    y: target vector
    r: simplex radius (1.0)
    Prox: Euclidean projection onto simplex via sorting
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF class
# ============================================================================


class SimplexIndicator(admm.UDFBase):
    def __init__(self, arg, radius=1.0):
        self.arg = arg
        self.radius = radius

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        if np.min(x) >= -1e-9 and abs(np.sum(x) - self.radius) <= 1e-9:
            return 0.0
        return float("inf")

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        sorted_v = np.sort(v)[::-1]
        cumulative = np.cumsum(sorted_v) - self.radius
        indices = np.arange(1, len(v) + 1)
        rho = np.nonzero(sorted_v - cumulative / indices > 0)[0][-1]
        theta = cumulative[rho] / (rho + 1)
        prox = np.maximum(v - theta, 0.0)
        return [prox.tolist()]


# ============================================================================
# Step 2: Generate data
# ============================================================================
y = np.array([0.2, -0.1, 0.7])
r = 1.0

print("=" * 70)
print("UDF Simplex Indicator Example")
print("=" * 70)
print(f"Target vector y: {y}")
print(f"Simplex radius r: {r}")
print(f"sum(y) = {np.sum(y):.4f}, min(y) = {np.min(y):.4f}")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", len(y))

# ============================================================================
# Step 4: Set objective
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + SimplexIndicator(x, r))

# ============================================================================
# Step 5: Solve the model
# ============================================================================
model.optimize()

# ============================================================================
# Step 6: Check status and print results
# ============================================================================
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"\u2713 Status: {model.StatusString}")
    print(f"\u2713 Optimal objective value: {model.ObjVal:.6f}")
    print(f"\u2713 Solver time: {model.SolverTime:.4f} seconds")
    print()

    x_sol = np.asarray(x.X)
    print(f"Solution x: {x_sol}")
    print()

    # Verify simplex constraints
    print("Constraint verification:")
    print(f"  sum(x) = {np.sum(x_sol):.6f} (should be {r})")
    print(f"  min(x) = {np.min(x_sol):.6f} (should be >= 0)")
    print(f"  Distance from y: {np.linalg.norm(x_sol - y):.6f}")

else:
    print(f"\u2717 Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
