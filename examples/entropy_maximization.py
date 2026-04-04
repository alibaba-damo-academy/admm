"""
Entropy Maximization Example

RUN COMMAND: python examples/entropy_maximization.py

This example demonstrates maximum entropy estimation using ADMM.

Problem: Find the probability distribution with maximum Shannon entropy
subject to linear equality and inequality constraints.

Mathematical formulation:
    min_x   sum_i x_i log(x_i)      (= -Shannon entropy)
    s.t.    A x = b
            F x <= g
            1^T x = 1               (probability simplex)
            x >= 0

Where:
    x: probability distribution (decision variable)
    A, b: linear equality constraints
    F, g: linear inequality constraints
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
n = 12      # number of outcomes
m = 3       # number of equality constraints
p = 2       # number of inequality constraints

# Generate a feasible point x0 (probability distribution)
x0 = np.random.rand(n)
x0 = x0 / np.sum(x0)

# Equality constraints: A x = b
A = np.random.randn(m, n)
b = A @ x0

# Inequality constraints: F x <= g
F = np.random.randn(p, n)
g = F @ x0 + np.random.rand(p)     # slack ensures feasibility

print("=" * 70)
print("Entropy Maximization Example")
print("=" * 70)
print(f"Number of outcomes (n): {n}")
print(f"Equality constraints (m): {m}")
print(f"Inequality constraints (p): {p}")
print(f"Uniform distribution entropy: {np.log(n):.6f}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Minimize sum(x_i * log(x_i)) = maximize Shannon entropy
model.setObjective(admm.sum(admm.entropy(x)))

# ============================================================================
# Step 4: Add constraints
# ============================================================================
model.addConstr(A @ x == b)          # linear equality constraints
model.addConstr(F @ x <= g)          # linear inequality constraints
model.addConstr(admm.sum(x) == 1)   # probability simplex
model.addConstr(x >= 0)             # non-negativity

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

    x_val = np.asarray(x.X)

    # Shannon entropy = -sum(x * log(x))
    entropy_val = -model.ObjVal
    print("Distribution statistics:")
    print(f"  Shannon entropy: {entropy_val:.6f}")
    print(f"  Maximum possible entropy (uniform): {np.log(n):.6f}")
    print(f"  Entropy ratio: {entropy_val / np.log(n):.4f}")
    print()

    # Distribution properties
    print(f"  Sum of probabilities: {np.sum(x_val):.6f} (should be 1.0)")
    print(f"  Min probability: {x_val.min():.6f}")
    print(f"  Max probability: {x_val.max():.6f}")
    print(f"  Nonzero probabilities (>1e-4): {np.sum(x_val > 1e-4)}/{n}")
    print()

    # Constraint verification
    eq_residual = np.linalg.norm(A @ x_val - b)
    ineq_violation = np.maximum(F @ x_val - g, 0).max()
    print("Constraint verification:")
    print(f"  Equality constraint residual: {eq_residual:.8f}")
    print(f"  Max inequality violation: {ineq_violation:.8f}")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
