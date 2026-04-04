"""
Linear Program Example

RUN COMMAND: python examples/linear_program.py

This example demonstrates a standard linear program using ADMM.

Mathematical formulation:
    min_x   c^T x
    s.t.    A x <= b
            x >= 0

Where:
    m = 15 inequality constraints, n = 10 variables, seed = 1.
    Problem constructed from a primal-dual feasible point to guarantee
    boundedness.
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate data from a known feasible point
# ============================================================================
m = 15
n = 10
s0 = np.random.randn(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.maximum(np.random.randn(n), 0)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

print("=" * 70)
print("Linear Program Example")
print("=" * 70)
print(f"Variables: {n}, Constraints: {m}")
print()

# ============================================================================
# Step 2: Create model and variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)

# ============================================================================
# Step 3: Set objective and constraints
# ============================================================================
model.setObjective(c.T @ x)
model.addConstr(A @ x <= b)
model.addConstr(x >= 0)

# ============================================================================
# Step 4: Solve
# ============================================================================
model.optimize()

# ============================================================================
# Step 5: Results
# ============================================================================
print()
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status: {model.StatusString}")
    print(f"  Objective value: {model.ObjVal:.6f}")
    print(f"  Solution norm: {np.linalg.norm(np.asarray(x.X)):.6f}")
else:
    print(f"  Optimization failed: {model.StatusString}")

print("=" * 70)
