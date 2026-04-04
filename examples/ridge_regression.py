"""
Ridge Regression Example

RUN COMMAND: python examples/ridge_regression.py

Ridge regression (L2-regularized least squares).

Mathematical formulation:
    min_beta  ||X beta - y||_2^2 + lam * ||beta||_2^2

Where:
    X    : design matrix (m x n)
    y    : response vector (m), generated as X beta_true + 0.5 * noise
    beta : regression coefficients to estimate (n)
    lam  : regularization parameter (1.0)
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
np.random.seed(1)

m = 100  # samples
n = 25   # features

X = np.random.randn(m, n)
beta_true = np.random.randn(n)
y = X @ beta_true + 0.5 * np.random.randn(m)  # noisy observations
lam = 1.0

print("=" * 70)
print("Ridge Regression Example")
print("=" * 70)
print(f"Samples (m):                {m}")
print(f"Features (n):               {n}")
print("Noise level:                0.5")
print(f"Regularization (lam):       {lam}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
beta = admm.Var("beta", n)

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Minimize: ||X beta - y||_2^2 + lam * ||beta||_2^2
model.setObjective(
    admm.sum(admm.square(X @ beta - y))
    + lam * admm.sum(admm.square(beta))
)

# ============================================================================
# Step 4: Solve the model
# ============================================================================
model.optimize()

# ============================================================================
# Step 5: Check status and print results
# ============================================================================
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status:    {model.StatusString}")
    print(f"  Objective: {model.ObjVal:.6f}")
    beta_est = np.asarray(beta.X)
    estimation_error = np.linalg.norm(beta_est - beta_true)
    print(f"  ||beta - beta_true||: {estimation_error:.6f}")
    print(f"  beta (first 5): {np.round(beta_est[:5], 6)}")
else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
