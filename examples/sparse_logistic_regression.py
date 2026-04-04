"""
Sparse Logistic Regression Example

RUN COMMAND: python examples/sparse_logistic_regression.py

Sparse logistic regression with L1 regularization for binary classification.
An intercept term v is included so that the sparsity penalty applies only to
the feature weights w.

Mathematical formulation:
    min_{w, v}   (1/m) sum_i log(1 + exp(-y_i (x_i^T w + v)))
                 + lam * ||w||_1

Where:
    w: feature weights (n-vector)
    v: intercept (scalar)
    X: design matrix  (m x n)
    y: labels in {-1, +1}
    lam: L1 regularization weight
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic classification data
# ============================================================================
n = 20
m = 60

beta = np.random.randn(n)
X = np.random.randn(m, n)
y = np.sign(X @ beta + 0.5 * np.random.randn(m))
y[y == 0] = 1  # ensure labels are +1 or -1

lam = 0.1

print("=" * 70)
print("Sparse Logistic Regression Example")
print("=" * 70)
print(f"Features    (n): {n}")
print(f"Samples     (m): {m}")
print(f"Regularization (lam): {lam}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
w = admm.Var("w", n)   # feature weights
v = admm.Var("v")       # intercept

# ============================================================================
# Step 3: Set objective -- logistic loss + L1 penalty on w only
# ============================================================================
margin = -y * (X @ w + v)
model.setObjective(admm.sum(admm.logistic(margin, 1)) / m + lam * admm.norm(w, ord=1))

# ============================================================================
# Step 4: Solve the model (no explicit constraints)
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
    print(f"  Number of non-zero coefficients: {np.sum(np.abs(np.array(w.X)) > 1e-6)}")
    print(f"  Intercept value: {float(v.X):.6f}")
else:
    print(f"  Optimization failed: {model.StatusString}")

print("=" * 70)
