"""
Second-Order Cone Program (SOCP) Example

RUN COMMAND: python examples/second_order_cone_program.py

This example demonstrates how to solve a Second-Order Cone Program using ADMM.

Problem: Minimize a linear objective subject to second-order cone constraints
and equality constraints.

Mathematical formulation:
    min_x   f^T x
    s.t.    ||A_i x + b_i||_2 <= c_i^T x + d_i,  i = 1, ..., m
            F x = g

Where:
    x: decision variable (n-dimensional)
    f: linear objective coefficients
    A_i, b_i, c_i, d_i: define SOC constraints
    F, g: define equality constraints
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
m = 3       # number of cone constraints
n = 10      # dimension of decision variable
p = 5       # number of equality constraints
n_i = 5     # dimension of each cone

f = np.random.randn(n)
A = []
b = []
c = []
d = []
x0 = np.random.randn(n)
for i in range(m):
    A.append(np.random.randn(n_i, n))
    b.append(np.random.randn(n_i))
    c.append(np.random.randn(n))
    d.append(np.linalg.norm(A[i] @ x0 + b[i], 2) - c[i].T @ x0)
F = np.random.randn(p, n)
g = F @ x0

print("=" * 70)
print("Second-Order Cone Program (SOCP) Example")
print("=" * 70)
print(f"Decision variable dimension (n): {n}")
print(f"Number of SOC constraints (m): {m}")
print(f"Cone dimension (n_i): {n_i}")
print(f"Number of equality constraints (p): {p}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)

# ============================================================================
# Step 3: Set objective
# ============================================================================
model.setObjective(f.T @ x)

# ============================================================================
# Step 4: Add constraints
# ============================================================================
# Second-order cone constraints: ||A_i x + b_i||_2 <= c_i^T x + d_i
for i in range(m):
    model.addConstr(admm.norm(A[i] @ x + b[i], ord=2) <= c[i].T @ x + d[i])

# Equality constraints: F x = g
model.addConstr(F @ x == g)

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
    print(f"  Status: {model.StatusString}")
    print(f"  Optimal objective value: {model.ObjVal:.6f}")
    print(f"  Solver time: {model.SolverTime:.4f} seconds")
    print()

    # Verify SOC constraints
    print("SOC constraint verification:")
    for i in range(m):
        lhs = np.linalg.norm(A[i] @ x.X + b[i], 2)
        rhs = c[i].T @ x.X + d[i]
        satisfied = lhs <= rhs + 1e-4
        mark = "ok" if satisfied else "VIOLATED"
        gap = rhs - lhs
        print(f"  {mark} Cone {i}: gap = {gap:.2e}")

    # Verify equality constraints
    eq_residual = np.linalg.norm(F @ x.X - g)
    print(f"\n  Equality constraint residual: {eq_residual:.8f}")

    # Solution statistics
    print(f"\n  Solution norm: {np.linalg.norm(x.X):.6f}")
    print(f"  Solution range: [{x.X.min():.6f}, {x.X.max():.6f}]")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
