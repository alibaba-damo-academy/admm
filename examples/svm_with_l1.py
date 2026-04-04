"""
SVM with L1 Regularization Example

RUN COMMAND: python examples/svm_with_l1.py

This example demonstrates a support vector machine with L1 regularization
for simultaneous classification and feature selection.

Mathematical formulation:
    min_{beta, v}   (1/m) sum_i max(0, 1 - y_i (x_i^T beta - v))
                  + lam * ||beta||_1

Where:
    beta: coefficient vector (n = 25, last 15 are zero in true model)
    v: intercept
    y_i in {-1, +1}: labels
    lam = 0.1: regularization weight
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
m = 120  # number of samples
n = 25   # number of features
beta_true = np.random.randn(n)
beta_true[10:] = 0  # last 15 coefficients are zero (sparse ground truth)
X = np.random.randn(m, n)
y = np.sign(X @ beta_true + 0.5 * np.random.randn(m))
y[y == 0] = 1
lam = 0.1

print("=" * 70)
print("SVM with L1 Regularization Example")
print("=" * 70)
print(f"Samples: {m}, Features: {n}")
print(f"True nonzero coefficients: {np.count_nonzero(beta_true)}")
print(f"Regularization (lam): {lam}")
print()

# ============================================================================
# Step 2: Create model and variables
# ============================================================================
model = admm.Model()
model.setOption(admm.Options.admm_max_iteration, 10000)
beta = admm.Var("beta", n)
v = admm.Var("v")

# ============================================================================
# Step 3: Set objective — hinge loss + L1 regularization
# ============================================================================
margin_loss = admm.sum(admm.maximum(1 - y * (X @ beta - v), 0))
model.setObjective(margin_loss / m + lam * admm.norm(beta, ord=1))

# ============================================================================
# Step 4: Solve
# ============================================================================
model.optimize()

# ============================================================================
# Step 5: Results
# ============================================================================
print()
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status: {model.StatusString}")
    print(f"  Objective value: {model.ObjVal:.6f}")
    beta_val = np.asarray(beta.X)
    nnz = np.count_nonzero(np.abs(beta_val) > 1e-6)
    print(f"  Nonzero coefficients: {nnz} (true: {np.count_nonzero(beta_true)})")
else:
    print(f"  Optimization failed: {model.StatusString}")

print("=" * 70)
