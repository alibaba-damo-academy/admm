"""
LASSO Regression Example

RUN COMMAND: python examples/lasso_regression.py

This example demonstrates LASSO (Least Absolute Shrinkage and Selection
Operator) regression using ADMM with a ``Param`` for the regularization
weight so the same symbolic model can be solved twice with different
penalty strengths.

Mathematical formulation:
    min_x  (1/2) * sum(square(A @ x - b)) + lam * ||x||_1

Where:
    A   : design matrix (m x n)
    b   : observation vector (m)
    x   : parameter vector to estimate (n)
    lam : regularization parameter (Param, solved for 0.05 and 0.2)
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
np.random.seed(1)

m = 30  # samples
n = 10  # features

A = np.random.randn(m, n)  # Data matrix
b = np.random.randn(m)     # Observations

print("=" * 70)
print("LASSO Regression Example")
print("=" * 70)
print(f"Samples (m): {m}")
print(f"Features (n): {n}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)          # Regression coefficients
lam = admm.Param("lam")       # Regularization weight as a parameter

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Minimize: (1/2) sum(square(A x - b)) + lam * ||x||_1
model.setObjective(
    0.5 * admm.sum(admm.square(A @ x - b))
    + lam * admm.norm(x, ord=1)
)

# ============================================================================
# Step 4: Set solver options
# ============================================================================
model.setOption(admm.Options.admm_max_iteration, 5000)

# ============================================================================
# Step 5: Solve with lam = 0.05 (light regularization)
# ============================================================================
model.optimize({"lam": 0.05})
x_small_penalty = np.asarray(x.X).copy()

print("Solve 1 (lam = 0.05):")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status:       {model.StatusString}")
    print(f"  Objective:    {model.ObjVal:.6f}")
    print(f"  ||x||_1:      {np.linalg.norm(x_small_penalty, 1):.6f}")
    print(f"  x = {np.round(x_small_penalty, 6)}")
else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print()

# ============================================================================
# Step 6: Solve with lam = 0.2 (heavier regularization)
# ============================================================================
model.optimize({"lam": 0.2})
x_large_penalty = np.asarray(x.X).copy()

print("Solve 2 (lam = 0.2):")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status:       {model.StatusString}")
    print(f"  Objective:    {model.ObjVal:.6f}")
    print(f"  ||x||_1:      {np.linalg.norm(x_large_penalty, 1):.6f}")
    print(f"  x = {np.round(x_large_penalty, 6)}")
else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print()

# ============================================================================
# Comparison
# ============================================================================
print("Comparison:")
print("-" * 70)
print(f"  ||x||_1 (lam=0.05): {np.linalg.norm(x_small_penalty, 1):.6f}")
print(f"  ||x||_1 (lam=0.2):  {np.linalg.norm(x_large_penalty, 1):.6f}")
print(f"  Heavier penalty produces sparser solution: "
      f"{np.linalg.norm(x_large_penalty, 1) < np.linalg.norm(x_small_penalty, 1)}")
print("=" * 70)
