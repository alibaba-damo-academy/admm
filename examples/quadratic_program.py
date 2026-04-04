"""
Quadratic Programming Example

RUN COMMAND: python examples/quadratic_program.py

Quadratic program with both inequality and equality constraints.

Mathematical formulation:
    min_x  (1/2) x^T P x + q^T x
    s.t.   G x <= h
           A x == b

Where:
    n = 10 variables
    m = 15 inequality constraints
    p = 5  equality constraints
    P = R^T R  (positive semidefinite)
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Generate problem data
# ============================================================================
np.random.seed(1)

m = 15  # inequality constraints
n = 10  # variables
p = 5   # equality constraints

P = np.random.randn(n, n)
P = P.T @ P                     # PSD via R^T R
q = np.random.randn(n)
G = np.random.randn(m, n)
h = G @ np.random.randn(n)      # feasible RHS
A = np.random.randn(p, n)
b = np.random.randn(p)

print("=" * 70)
print("Quadratic Programming Example")
print("=" * 70)
print(f"Variables (n):              {n}")
print(f"Inequality constraints (m): {m}")
print(f"Equality constraints (p):   {p}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)

# ============================================================================
# Step 3: Set objective
# ============================================================================
model.setObjective(0.5 * x.T @ P @ x + q.T @ x)

# ============================================================================
# Step 4: Add constraints
# ============================================================================
model.addConstr(G @ x <= h)   # inequality constraints
model.addConstr(A @ x == b)   # equality constraints

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
    print(f"  Status:    {model.StatusString}")
    print(f"  Objective: {model.ObjVal:.6f}")
    print(f"  x = {np.round(np.asarray(x.X), 6)}")
else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
