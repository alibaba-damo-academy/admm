"""
Overview Portfolio Example (Chapter 1 Illustrative Example)

RUN COMMAND: python examples/overview_portfolio.py

This is the introductory portfolio example from the ADMM documentation overview.

Mathematical formulation:
    min_w   -mu^T w + gamma * w^T Sigma w
    s.t.    sum(w) = 1
            w >= 0

Where:
    w: portfolio weights (n = 20 assets)
    mu: expected returns
    Sigma: covariance matrix (factor model, PSD)
    gamma: risk-aversion parameter
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
n = 20
mu = np.abs(np.random.randn(n))
Sigma = np.random.randn(n + 3, n)
Sigma = Sigma.T @ Sigma + 0.1 * np.eye(n)
gamma = 0.5

print("=" * 70)
print("Overview Portfolio Example (n=20, seed=1)")
print("=" * 70)
print(f"Number of assets: {n}")
print(f"Risk-aversion parameter (gamma): {gamma}")
print()

# ============================================================================
# Step 2: Create model and variables
# ============================================================================
model = admm.Model()
w = admm.Var("w", n)

# ============================================================================
# Step 3: Set objective
# ============================================================================
model.setObjective(-mu.T @ w + gamma * (w.T @ Sigma @ w))

# ============================================================================
# Step 4: Add constraints
# ============================================================================
model.addConstr(admm.sum(w) == 1)
model.addConstr(w >= 0)

# ============================================================================
# Step 5: Solve
# ============================================================================
model.optimize()

# ============================================================================
# Step 6: Results
# ============================================================================
print()
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status: {model.StatusString}")
    print(f"  Objective value: {model.ObjVal:.6f}")      # Expected: ~ -1.08
    print(f"  Sum of weights: {np.sum(np.asarray(w.X)):.6f}")
    print(f"  Min weight: {np.min(np.asarray(w.X)):.6f}")
else:
    print(f"  Optimization failed: {model.StatusString}")

print("=" * 70)
