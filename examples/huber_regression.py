"""
Huber Regression Example

RUN COMMAND: python examples/huber_regression.py

Huber regression - robust regression that is less sensitive to outliers
than ordinary least squares.  The Huber loss is quadratic for small
residuals and linear for large ones, so outliers receive bounded influence.

Mathematical formulation:
    min_beta   sum_i  huber( (X beta - y)_i , M )

Where:
    beta: regression coefficients (n-vector)
    X: design matrix  (m x n)
    y: observations   (m-vector, first 8 entries are outliers)
    M: Huber threshold (M = 1.0)
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data with outliers
# ============================================================================
m = 80
n = 20

X = np.random.randn(m, n)
beta_true = np.random.randn(n)
y = X @ beta_true + 0.1 * np.random.randn(m)
y[:8] += 8.0 * np.random.randn(8)  # inject outliers into first 8 observations

print("=" * 70)
print("Huber Regression Example")
print("=" * 70)
print(f"Observations (m): {m}")
print(f"Features     (n): {n}")
print("Outlier observations: first 8 (shifted by 8.0 * noise)")
print()

# ============================================================================
# Step 2: Create model and decision variable
# ============================================================================
model = admm.Model()
beta = admm.Var("beta", n)

# ============================================================================
# Step 3: Set objective -- Huber loss with threshold M = 1.0
# ============================================================================
residual = X @ beta - y
model.setObjective(admm.sum(admm.huber(residual, 1.0)))

# ============================================================================
# Step 4: Solve the model (no constraints needed)
# ============================================================================
model.optimize()

# ============================================================================
# Step 5: Check status and print results
# ============================================================================
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status: {model.StatusString}")
    print(f"  Optimal objective value: {model.ObjVal:.6f}")
    print(f"  Estimation error: {np.linalg.norm(np.array(beta.X) - beta_true):.6f}")
else:
    print(f"  Optimization failed: {model.StatusString}")

print("=" * 70)
