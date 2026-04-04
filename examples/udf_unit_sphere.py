"""
UDF Unit-Sphere Indicator Example

RUN COMMAND: python examples/udf_unit_sphere.py

This example demonstrates projection onto the unit sphere via a User-Defined
Function (UDF). The indicator enforces ||x||_2 = 1 (nonconvex constraint)
and its proximal operator normalizes the input vector.

Problem: Find the closest point x on the unit sphere to a given vector y.

Mathematical formulation:
    min_x   (1/2) ||x - y||_2^2  +  delta_{||x||_2 = 1}(x)

Where:
    delta is the indicator function (0 if ||x||=1, +inf otherwise)
    y: target vector
    Prox: x = v / ||v||_2  (for v != 0)
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF class
# ============================================================================


class UnitSphereIndicator(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        norm = np.linalg.norm(x)
        return 0.0 if abs(norm - 1.0) <= 1e-9 else float("inf")

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        norm = np.linalg.norm(v)
        if norm <= 1e-12:
            prox = np.zeros_like(v)
            prox[0] = 1.0
            return [prox.tolist()]
        return [(v / norm).tolist()]


# ============================================================================
# Step 2: Generate data
# ============================================================================
y = np.array([0.1, 0.0])

print("=" * 70)
print("UDF Unit-Sphere Indicator Example")
print("=" * 70)
print(f"Target vector y: {y}")
print(f"||y||_2 = {np.linalg.norm(y):.6f}")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", 2)

# ============================================================================
# Step 4: Set objective
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + UnitSphereIndicator(x))

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
    print(f"||x||_2 = {np.linalg.norm(x_sol):.6f} (should be 1.0)")
    print(f"Distance from y: {np.linalg.norm(x_sol - y):.6f}")

else:
    print(f"\u2717 Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
