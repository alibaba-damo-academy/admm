"""
Portfolio Optimization Example

RUN COMMAND: python examples/portfolio_optimization.py

This example demonstrates mean-variance portfolio optimization using ADMM.

Problem: Allocate budget across n assets to maximize expected return while
controlling risk (variance of returns).

Mathematical formulation:
    min    -expected_return + gamma * risk
    s.t.   sum(w) = 1          (fully invested)
           w >= 0              (long-only, no short-selling)

Where:
    w: portfolio weights (decision variable)
    expected_return = mu^T w
    risk = w^T Sigma w
    mu: expected returns of assets
    Sigma: covariance matrix of asset returns
    gamma: risk-aversion parameter (larger = more conservative)
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
n = 50  # number of assets

# Expected returns (positive, random)
mu = np.abs(np.random.randn(n))

# Covariance matrix (positive semi-definite via factor model)
F = np.random.randn(n + 5, n)
Sigma = F.T @ F + 0.1 * np.eye(n)

# Risk-aversion parameter
gamma = 0.5

print("=" * 70)
print("Portfolio Optimization Example")
print("=" * 70)
print(f"Number of assets: {n}")
print(f"Risk-aversion parameter (gamma): {gamma}")
print(f"Expected return range: [{mu.min():.3f}, {mu.max():.3f}]")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
w = admm.Var("w", n)  # portfolio weights

# ============================================================================
# Step 3: Set objective using intermediate variables
# ============================================================================
# Maximize return (minimize -mu^T w) while minimizing risk (w^T Sigma w)
expected_return = mu.T @ w
risk = w.T @ Sigma @ w
model.setObjective(-expected_return + gamma * risk)

# ============================================================================
# Step 4: Add constraints
# ============================================================================
# Fully invested: sum of weights = 1
model.addConstr(admm.sum(w) == 1)

# Long-only: all weights >= 0
model.addConstr(w >= 0)

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
    print()

    # Portfolio statistics
    portfolio_return = mu.T @ np.array(w.X)
    portfolio_risk = np.sqrt(np.array(w.X).T @ Sigma @ np.array(w.X))
    print(f"  Portfolio expected return: {portfolio_return:.6f}")
    print(f"  Portfolio risk (std dev):  {portfolio_risk:.6f}")
    print()

    # Show top 5 holdings
    w_arr = np.array(w.X)
    top_5_idx = np.argsort(w_arr)[-5:][::-1]
    print("  Top 5 holdings:")
    for i, idx in enumerate(top_5_idx, 1):
        print(f"    {i}. Asset {idx}: {w_arr[idx]*100:.2f}%")

    print()
    print(f"  Number of assets with >1% allocation: {np.sum(w_arr > 0.01)}")

    # Verify constraints
    print()
    print("  Constraint verification:")
    print(f"    Sum of weights: {np.sum(w_arr):.6f} (should be 1.0)")
    print(f"    Minimum weight: {w_arr.min():.6f}")
else:
    print(f"  Optimization failed: {model.StatusString}")

print("=" * 70)
