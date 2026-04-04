"""
Quantile Regression Example

RUN COMMAND: python examples/quantile_regression.py

This example demonstrates quantile regression using the pinball loss with ADMM.

Problem: Estimate the conditional quantile of a response variable, which is
more robust to outliers than ordinary least squares for non-symmetric
error distributions.

Mathematical formulation:
    min_w   (1/2) ||X w - y||_1 + (1/2 - tau) * 1^T (X w - y)

Where:
    w: regression coefficients (decision variable)
    X: design matrix (m x n)
    y: response vector (m-dimensional)
    tau: quantile level (0 < tau < 1)

The pinball loss decomposes as:
    rho_tau(u) = tau * max(u, 0) + (1-tau) * max(-u, 0)
              = (1/2) |u| + (1/2 - tau) * u
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
n = 10      # number of features
m = 200     # number of observations

beta = np.random.randn(n)
X = np.random.randn(m, n)
y = X @ beta + 0.5 * np.random.randn(m)    # moderate noise
tau = 0.9                                    # 90th percentile

print("=" * 70)
print("Quantile Regression Example")
print("=" * 70)
print(f"Number of features (n): {n}")
print(f"Number of observations (m): {m}")
print(f"Quantile level (tau): {tau}")
print("Noise level: 0.5")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
model.setOption(admm.Options.admm_max_iteration, 10000)
model.setOption(admm.Options.termination_absolute_error_threshold, 1e-5)
model.setOption(admm.Options.termination_relative_error_threshold, 1e-5)
w = admm.Var("w", n)

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Pinball loss: (1/2)||r||_1 + (1/2 - tau)*sum(r)
residual = X @ w - y
model.setObjective(0.5 * admm.norm(residual, ord=1) + (0.5 - tau) * admm.sum(residual))

# ============================================================================
# Step 4: Add constraints
# ============================================================================
# Unconstrained problem (no additional constraints)

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

    # Regression statistics
    residuals = X @ w.X - y
    print("Coefficient statistics:")
    print(f"  Number of coefficients: {n}")
    print(f"  Nonzero coefficients (|w| > 1e-4): {np.sum(np.abs(w.X) > 1e-4)}")
    print(f"  Coefficient norm: {np.linalg.norm(w.X):.6f}")
    print()

    # Compare with true coefficients
    coef_error = np.linalg.norm(w.X - beta)
    print(f"  Coefficient recovery error (||w - beta_true||_2): {coef_error:.6f}")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
