"""
UDF Binary Indicator Example

RUN COMMAND: python examples/udf_binary.py

This example demonstrates binary constraint enforcement via a User-Defined
Function (UDF). The indicator enforces x in {0, 1}^n (nonconvex binary cube)
and its proximal operator rounds coordinatewise to the nearest binary value.

Problem: Find the closest binary vector x to a given vector y.

Mathematical formulation:
    min_x   (1/2) ||x - y||_2^2  +  delta_{{0,1}^n}(x)

Where:
    delta is the indicator function (0 if x in {0,1}^n, +inf otherwise)
    y: target vector
    Prox (coordinatewise): x_i = 0 if v_i < 0.5, else 1
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF class
# ============================================================================


class BinaryIndicator(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        is_binary = np.logical_or(np.abs(x) <= 1e-9, np.abs(x - 1.0) <= 1e-9)
        return 0.0 if np.all(is_binary) else float("inf")

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        prox = np.where(v >= 0.5, 1.0, 0.0)
        return [prox.tolist()]


# ============================================================================
# Step 2: Generate data
# ============================================================================
y = np.array([0.2, 0.8, 1.4, -0.3])

print("=" * 70)
print("UDF Binary Indicator Example")
print("=" * 70)
print(f"Target vector y: {y}")
print(f"Dimension: {len(y)}")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", len(y))

# ============================================================================
# Step 4: Set objective
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + BinaryIndicator(x))

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

    # Verify binary constraint and show rounding
    print("Rounding details:")
    for i in range(len(y)):
        print(f"  y[{i}] = {y[i]:+.1f}  ->  x[{i}] = {x_sol[i]:.0f}")

    print(f"\nAll binary: {np.all(np.logical_or(np.abs(x_sol) < 1e-6, np.abs(x_sol - 1.0) < 1e-6))}")
    print(f"Distance from y: {np.linalg.norm(x_sol - y):.6f}")

else:
    print(f"\u2717 Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
