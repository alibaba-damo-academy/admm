"""
User-Defined Smooth Function (grad UDF): Log-Cosh Robust Regression

RUN COMMAND: python examples/udf_grad_log_cosh.py

This example demonstrates the simplest grad-based UDF: the log-cosh loss.
Instead of deriving a proximal operator, we supply only eval() and grad(),
and the C++ backend solves the proximal subproblem automatically via
gradient descent with Armijo backtracking line search.

Problem: Robust regression using log-cosh loss, which is a smooth
approximation to L1 that behaves like L2 near zero and L1 for large
residuals.

Mathematical formulation:
    min_x   sum(log(cosh(A @ x - b))) + (lam/2) ||x||_2^2

Where:
    f(r) = log(cosh(r))  is the log-cosh loss
    grad f(r) = tanh(r)  is bounded in [-1, 1]
    The L2 term provides regularization

Comparison:
    - L2 loss:  penalizes outliers quadratically — sensitive to outliers
    - L1 loss:  constant gradient — robust but not smooth
    - Log-cosh: smooth everywhere, gradient tanh(r) → ±1 for large r — robust AND smooth
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define the Log-Cosh Loss as a grad-based UDF
# ============================================================================
class LogCoshLoss(admm.UDFBase):
    """Log-cosh loss: f(r) = sum(log(cosh(r_i))).

    This is a smooth approximation to the absolute value:
        log(cosh(r)) ≈ |r| - log(2)   for large |r|
        log(cosh(r)) ≈ r^2 / 2        for small |r|

    Gradient: tanh(r), bounded in [-1, 1].
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.log(np.cosh(r))))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [np.tanh(r)]


# ============================================================================
# Step 2: Generate data with outliers
# ============================================================================
np.random.seed(42)
n, p = 50, 5
A = np.random.randn(n, p)
x_true = np.array([1.0, -2.0, 0.5, 0.0, 1.5])
b = A @ x_true + 0.1 * np.random.randn(n)

# Inject 20% outliers
outlier_idx = np.random.choice(n, size=10, replace=False)
b[outlier_idx] += np.random.choice([-1, 1], size=10) * np.random.uniform(8, 15, size=10)

lam = 0.1

print("=" * 70)
print("Grad UDF: Log-Cosh Robust Regression")
print("=" * 70)
print(f"Data: {n} samples, {p} features, {len(outlier_idx)} outliers")
print(f"True coefficients: {x_true}")
print(f"Regularization (lam): {lam}")
print()

# ============================================================================
# Step 3: Solve with ADMM using the grad-based UDF
# ============================================================================
model = admm.Model()
x = admm.Var("x", p)
residual = A @ x - b

model.setObjective(LogCoshLoss(residual) + (lam / 2) * admm.sum(admm.square(x)))
model.optimize()

print("Results:")
print("-" * 70)
print(f"  Status: {model.StatusString}")
print(f"  Objective: {model.ObjVal:.6f}")
print()

x_val = np.asarray(x.X)
print("  Coefficient comparison:")
print(f"  {'Index':<8} {'True':<12} {'Recovered':<12} {'Error':<12}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
for i in range(p):
    print(f"  {i:<8} {x_true[i]:<12.4f} {x_val[i]:<12.4f} {abs(x_val[i] - x_true[i]):<12.4f}")

print(f"\n  ||x - x_true||_2 = {np.linalg.norm(x_val - x_true):.6f}")

# ============================================================================
# Step 4: Compare with ordinary least squares (sensitive to outliers)
# ============================================================================
x_ols = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"\n  OLS error (no robustness):  ||x_ols - x_true||_2 = {np.linalg.norm(x_ols - x_true):.6f}")
print(f"  Log-cosh error (robust):   ||x_udf - x_true||_2 = {np.linalg.norm(x_val - x_true):.6f}")

print("=" * 70)
